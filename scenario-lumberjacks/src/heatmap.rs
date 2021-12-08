use std::collections::BTreeMap;
use std::{f32, fs};

use ggez::graphics;
use ggez::graphics::{Canvas, Color, DrawMode, Mesh, Rect};
use image::png::PngEncoder;
use image::{ColorType, ImageBuffer, Rgba};
use npc_engine_turn::SnapshotDiffRef;

use crate::{output_path, PostMCTSHookArgs, PostMCTSHookFn, SPRITE_SIZE, StateRef, State};

// TODO
pub fn heatmap_hook() -> PostMCTSHookFn {
    Box::new(
        move |PostMCTSHookArgs {
                  ctx,
                  assets,
                  run,
                  turn,
                  world,
                  agent,
                  mcts,
                  ..
              }| {
            if let Some(ctx) = ctx {
                struct HeatMapEntry {
                    visits: usize,
                    score: f32,
                }

                let mut positions: BTreeMap<(isize, isize), HeatMapEntry> = BTreeMap::new();
                let mut max_visits = 0;
                let mut best_avg_score: f32 = 0.;
                let mut worst_avg_score: f32 = f32::MAX;

                // Get nodes for this agent
                mcts.nodes().for_each(|(_, edges)| {
                    edges.into_iter().for_each(|(_, edge)| {
                        let edge = edge.borrow();
                        let child = edge.child.upgrade().unwrap();

                        if child.agent() == agent {
                            let snapshot = StateRef::Snapshot(
                                SnapshotDiffRef::new(&mcts.snapshot, child.diff())
                            );

                            let (x, y) = snapshot.find_agent(mcts.agent()).unwrap();

                            let visits = edge.visits;
                            let score = edge.q_values.get(&mcts.agent()).copied().unwrap_or(0.);
                            let entry = positions
                                .entry((x, y))
                                .and_modify(|entry| {
                                    entry.visits += visits;
                                    entry.score += score;
                                })
                                .or_insert(HeatMapEntry { visits, score });

                            best_avg_score = best_avg_score.max(entry.score / entry.visits as f32);
                            worst_avg_score =
                                worst_avg_score.min(entry.score / entry.visits as f32);
                            max_visits = max_visits.max(edge.visits);
                        }
                    })
                });

                // Heatmap
                {
                    let canvas = Canvas::with_window_size(ctx).unwrap();
                    graphics::set_canvas(ctx, Some(&canvas));
                    graphics::clear(ctx, graphics::BLACK);

                    let rect = Mesh::new_rectangle(
                        ctx,
                        DrawMode::fill(),
                        Rect::new(0 as f32, 0 as f32, SPRITE_SIZE, SPRITE_SIZE),
                        graphics::WHITE,
                    )
                    .unwrap();

                    world.with_map_coordinates(ctx, |ctx| {
                        for (&(x, y), entry) in &positions {
                            let visits = entry.visits as f32 / max_visits as f32;

                            // Cull visits that are not significant to avoid outliers
                            if visits < 0.001 {
                                continue;
                            }

                            let scores = (entry.score / entry.visits as f32 - worst_avg_score)
                                / (best_avg_score - worst_avg_score + f32::EPSILON);

                            let mut green = scores;
                            let mut red = 1. - scores;
                            let max = red.max(green);

                            // Normalize to max
                            green /= max;
                            red /= max;

                            if visits > f32::EPSILON && scores > f32::EPSILON {
                                graphics::draw(
                                    ctx,
                                    &rect,
                                    (
                                        [x as f32 * SPRITE_SIZE, y as f32 * SPRITE_SIZE],
                                        Color::new(red, green, 0., visits),
                                    ),
                                )
                                .unwrap();
                            }
                        }
                    });

                    world.draw(ctx, assets);

                    graphics::present(ctx).unwrap();
                    let image = canvas.image();

                    let width = image.width() as u32;
                    let height = image.height() as u32;
                    let image_data: ImageBuffer<Rgba<u8>, _> =
                        ImageBuffer::from_raw(width, height, image.to_rgba8(ctx).unwrap()).unwrap();

                    let flipped_image_data = image::imageops::flip_vertical(&image_data);

                    fs::create_dir_all(format!(
                        "{}/{}/heatmaps/agent{}/",
                        output_path(),
                        run.map(|n| n.to_string()).unwrap_or_default(),
                        agent.0
                    ))
                    .unwrap();
                    let file = fs::OpenOptions::new()
                        .create(true)
                        .write(true)
                        .truncate(true)
                        .open(format!(
                            "{}/{}/heatmaps/agent{}/{:06}.png",
                            output_path(),
                            run.map(|n| n.to_string()).unwrap_or_default(),
                            agent.0,
                            turn
                        ))
                        .unwrap();

                    PngEncoder::new(file)
                        .encode(&flipped_image_data, width, height, ColorType::Rgba8)
                        .unwrap();
                }
            }
        },
    )
}
