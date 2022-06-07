/* 
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::{char, num::NonZeroU8};
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::mem;
use std::{fmt, io};

use ggez::graphics::{draw, Image, WHITE};
use ggez::Context;
use image::png::PngDecoder;
use image::{ColorType, ImageDecoder};
use npc_engine_common::AgentId;

use serde::Serialize;

use crate::{config, Action, SPRITE_SIZE};

#[derive(Clone, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct TileMap {
    pub width: usize,
    pub height: usize,
    pub tiles: Box<[Box<[Tile]>]>,
}

impl fmt::Debug for TileMap {
    fn fmt(&self, _: &mut fmt::Formatter) -> fmt::Result {
        todo!()
    }
}

impl TileMap {
    pub fn from_io(agents: &mut Vec<AgentId>, read: impl io::Read) -> Self {
        let decoder = PngDecoder::new(read).expect("failed to parse value as png");
        let (width, height) = decoder.dimensions();
        let width = width as usize;
        let height = height as usize;

        let tiles = match decoder.color_type() {
            ColorType::Rgba8 => {
                let mut data = vec![0; decoder.total_bytes() as _];
                decoder.read_image(&mut data).expect("failed to read image");

                data.chunks_exact(width * 4)
                    .map(|row| {
                        row.chunks_exact(4)
                            .map(|slice| match (slice[0], slice[1], slice[2], slice[3]) {
                                (0, 255, 0, 255) => Tile::Tree(config().map.tree_height),
                                (255, 255, 255, 255) => {
                                    let agent = AgentId(agents.len() as u32);
                                    agents.push(agent);
                                    Tile::Agent(agent)
                                }
                                (0, 0, 0, 255) => Tile::Impassable,
                                (0, 0, 255, 255) => Tile::Well,
                                (_, _, _, 0) => Tile::Empty,
                                (r, g, b, a) => panic!(
                                    "failed to parse color ({}, {}, {}, {}) from png",
                                    r, g, b, a
                                ),
                            })
                            .collect::<Vec<_>>()
                            .into_boxed_slice()
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            }
            _ => panic!("unsupported png color format"),
        };

        TileMap {
            width,
            height,
            tiles,
        }
    }

    #[allow(clippy::needless_borrow)] // because clippy is buggy in its analysis here
    pub fn draw(
        &self,
        ctx: &mut Context,
        assets: &BTreeMap<String, Image>,
        actions: &BTreeMap<AgentId, Action>,
    ) {
        self.tiles
            .iter()
            .enumerate()
            .flat_map(|(row_id, row)| {
                row.iter()
                    .enumerate()
                    .map(move |(col_id, tile)| (row_id, col_id, tile))
            })
            .for_each(|(row, col, tile)| {
                let sprite = match tile {
                    Tile::Agent(agent) if actions.contains_key(&agent) => Some(format!(
                        "{}{}",
                        if agent.0 % 2 == 0 { "Orange" } else { "Yellow" },
                        actions.get(&agent).unwrap().sprite_name(),
                    )),
                    tile => tile.sprite(),
                };

                if let Some(sprite) = sprite {
                    draw(
                        ctx,
                        assets.get(&sprite).unwrap(),
                        ([col as f32 * SPRITE_SIZE, row as f32 * SPRITE_SIZE], WHITE),
                    )
                    .unwrap();
                }
            });
    }

    pub fn tree_count(&self) -> usize {
        self.tiles
            .iter()
            .flat_map(|cols| {
                cols.iter().filter(|&&tile| {
                    matches!(tile, Tile::Tree(_))
                })
            })
            .count()
    }

    pub fn patch_count(&self, patch_size: usize) -> usize {
        // Set width/height to nearest multiple of patch size
        let width = self.width - self.width % patch_size;
        let height = self.height - self.height % patch_size;

        (0..height)
            .step_by(patch_size)
            .flat_map(|offset_y| {
                (0..width).step_by(patch_size).map(move |offset_x| {
                    (0..patch_size)
                        .fold(DefaultHasher::default(), |hasher, y| {
                            (0..patch_size).fold(hasher, |mut hasher, x| {
                                self.tiles[offset_y + y][offset_x + x].hash(&mut hasher);
                                hasher
                            })
                        })
                        .finish()
                })
            })
            .collect::<HashSet<_>>()
            .len()
    }
}

#[derive(Clone, Default, PartialEq, Eq)]
pub struct TileMapSnapshot {
    pub top: isize,
    pub left: isize,
    pub tiles: Box<[Box<[Tile]>]>,
}

impl fmt::Debug for TileMapSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut string = String::new();
        for cols in self.tiles.iter() {
            for tile in cols.iter() {
                string.push(match tile {
                    Tile::Empty => '+',
                    Tile::Agent(_) => '@',
                    Tile::Tree(height) => char::from_digit(height.get().into(), 10).unwrap(),
                    Tile::Barrier => 'B',
                    Tile::Impassable => 'X',
                    Tile::Well => 'W',
                });
            }
            string.push('\n');
        }

        write!(f, "{}", string.as_str())
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Tile {
    Tree(NonZeroU8),
    Agent(AgentId),
    Impassable,
    Barrier,
    Well,
    Empty,
}

impl Default for Tile {
    fn default() -> Self {
        Tile::Empty
    }
}

impl Tile {
    fn sprite(&self) -> Option<String> {
        match self {
            Tile::Tree(height) => Some(format!("Tree{}_3", height.get().min(3))),
            Tile::Agent(agent) => Some(if agent.0 % 2 == 0 {
                "OrangeRight".to_owned()
            } else {
                "YellowRight".to_owned()
            }),
            Tile::Barrier => Some("WoodenBarrier".to_owned()),
            Tile::Impassable => Some("ImpassableRock".to_owned()),
            Tile::Well => Some("Well".to_owned()),
            Tile::Empty => None,
        }
    }

    pub fn is_impassable(&self) -> bool {
        matches!(self, Tile::Impassable)
    }

    pub fn is_walkable(&self) -> bool {
        matches!(self, Tile::Empty)
    }

    pub fn is_pathfindable(&self) -> bool {
        matches!(self, Tile::Empty | Tile::Agent(_))
    }

    pub fn is_support(&self) -> bool {
        matches!(self, Tile::Impassable | Tile::Barrier)
    }

    pub fn is_point_of_interest(&self) -> bool {
        matches!(self, Tile::Tree(_) | Tile::Well)
    }
}

#[derive(Clone, Debug, Default, Eq)]
pub struct TileMapDiff {
    pub agents_pos: HashMap<AgentId, (isize, isize)>,
    pub tiles: BTreeMap<(isize, isize), Tile>,
}

impl Hash for TileMapDiff {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.tiles.hash(state);
    }
}

impl PartialEq for TileMapDiff {
    fn eq(&self, other: &Self) -> bool {
        self.tiles.eq(&other.tiles)
    }
}

impl TileMapDiff {
    pub fn diff_size(&self) -> usize {
        self.tiles.len() * mem::size_of::<(isize, isize, Tile)>()
    }
}
