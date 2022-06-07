/* 
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich, see AUTHORS.txt for details
 */

use std::{env, f32, fs};

use ggez::conf::{WindowMode, WindowSetup};
use ggez::graphics::Image;
use ggez::ContextBuilder;
use image::{png::PngDecoder, ImageDecoder};

use npc_engine_common::graphviz::GRAPH_OUTPUT_DEPTH;
use rand::{thread_rng, RngCore};
use rayon::prelude::*;

use lumberjacks::{batch, config, name, output_path, GameState, SPRITE_SIZE};

const ASSETS: &[(&str, &[u8])] = &[
    (
        "ImpassableRock",
        include_bytes!("../../assets/ImpassableRock.png"),
    ),
    ("OrangeDown", include_bytes!("../../assets/OrangeDown.png")),
    (
        "OrangeDownBarrier",
        include_bytes!("../../assets/OrangeDownBarrier.png"),
    ),
    (
        "OrangeDownChopping",
        include_bytes!("../../assets/OrangeDownChopping.png"),
    ),
    ("OrangeLeft", include_bytes!("../../assets/OrangeLeft.png")),
    (
        "OrangeLeftBarrier",
        include_bytes!("../../assets/OrangeLeftBarrier.png"),
    ),
    (
        "OrangeLeftChopping",
        include_bytes!("../../assets/OrangeLeftChopping.png"),
    ),
    (
        "OrangeRight",
        include_bytes!("../../assets/OrangeRight.png"),
    ),
    (
        "OrangeRightBarrier",
        include_bytes!("../../assets/OrangeRightBarrier.png"),
    ),
    (
        "OrangeRightChopping",
        include_bytes!("../../assets/OrangeRightChopping.png"),
    ),
    ("OrangeTop", include_bytes!("../../assets/OrangeTop.png")),
    (
        "OrangeTopBarrier",
        include_bytes!("../../assets/OrangeTopBarrier.png"),
    ),
    (
        "OrangeTopChopping",
        include_bytes!("../../assets/OrangeTopChopping.png"),
    ),
    ("Tree1_3", include_bytes!("../../assets/Tree1_3.png")),
    ("Tree2_3", include_bytes!("../../assets/Tree2_3.png")),
    ("Tree3_3", include_bytes!("../../assets/Tree3_3.png")),
    (
        "TreeSapling",
        include_bytes!("../../assets/TreeSapling.png"),
    ),
    (
        "WoodenBarrier",
        include_bytes!("../../assets/WoodenBarrier.png"),
    ),
    ("YellowDown", include_bytes!("../../assets/YellowDown.png")),
    (
        "YellowDownBarrier",
        include_bytes!("../../assets/YellowDownBarrier.png"),
    ),
    (
        "YellowDownChopping",
        include_bytes!("../../assets/YellowDownChopping.png"),
    ),
    ("YellowLeft", include_bytes!("../../assets/YellowLeft.png")),
    (
        "YellowLeftBarrier",
        include_bytes!("../../assets/YellowLeftBarrier.png"),
    ),
    (
        "YellowLeftChopping",
        include_bytes!("../../assets/YellowLeftChopping.png"),
    ),
    (
        "YellowRight",
        include_bytes!("../../assets/YellowRight.png"),
    ),
    (
        "YellowRightBarrier",
        include_bytes!("../../assets/YellowRightBarrier.png"),
    ),
    (
        "YellowRightChopping",
        include_bytes!("../../assets/YellowRightChopping.png"),
    ),
    ("YellowTop", include_bytes!("../../assets/YellowTop.png")),
    (
        "YellowTopBarrier",
        include_bytes!("../../assets/YellowTopBarrier.png"),
    ),
    (
        "YellowTopChopping",
        include_bytes!("../../assets/YellowTopChopping.png"),
    ),
    ("Well", include_bytes!("../../assets/Well.png")),
];

fn main() {
    env_logger::init();

    fs::create_dir_all(format!("{}/", &output_path())).unwrap();

    let info = serde_json::json!({
        "git-hash": env!("GIT_HASH"),
        "config": config(),
    });

    let file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(format!("{}/info.json", output_path(),))
        .unwrap();

    serde_json::to_writer_pretty(file, &info).unwrap();

    GRAPH_OUTPUT_DEPTH.store(config().analytics.graphs_depth, std::sync::atomic::Ordering::Relaxed);
    if batch() {
        (0..config().batch.runs).into_par_iter().for_each(|run| {
            let seed = config()
                .mcts
                .seed
                .unwrap_or_else(|| thread_rng().next_u64());
            let mut state = GameState::new(config().display.interactive, Some(run), seed);

            let turns = config()
                .turns
                .expect("Running batch mode with no turn limit!");

            state.dump_run();
            while state.turn() < turns {
                state.update(None);
            }
            state.dump_result();
        });
    } else {
        let seed = config()
            .mcts
            .seed
            .unwrap_or_else(|| thread_rng().next_u64());
        let mut state = GameState::new(config().display.interactive, None, seed);
        state.dump_run();

        // Create game context
        let (mut ctx, mut events) = ContextBuilder::new("lumberjacks", "Sven Knobloch")
            .window_setup(WindowSetup {
                title: name().clone(),
                vsync: true,
                ..Default::default()
            })
            .window_mode(WindowMode::default().dimensions(
                (2 * config().display.padding.0 + state.width()) as f32 * SPRITE_SIZE,
                (2 * config().display.padding.1 + state.height()) as f32 * SPRITE_SIZE,
            ))
            .build()
            .unwrap();

        // Load assets
        for (name, bytes) in ASSETS {
            let png = PngDecoder::new(<&[u8]>::clone(bytes)).unwrap();
            let (width, height) = png.dimensions();

            let mut rgba = vec![0u8; png.total_bytes() as _];

            png.read_image(&mut rgba).unwrap();

            state.add_asset(
                (*name).to_owned(),
                Image::from_rgba8(&mut ctx, width as _, height as _, &rgba).unwrap(),
            );
        }

        // Screenshot of initial state
        let dir = state.output_dir();
        state.screenshot(&mut ctx, &format!("{}/start.png", dir));

        // Run game
        ggez::event::run(&mut ctx, &mut events, &mut state).unwrap();

        // Screenshot of final state
        state.screenshot(&mut ctx, &format!("{}/result.png", dir));

        state.dump_result();
    }
}
