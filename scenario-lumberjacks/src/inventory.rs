use std::collections::BTreeMap;
use std::mem;

use ggez::graphics::{draw, Image, Text, DEFAULT_FONT_SCALE, WHITE};
use ggez::Context;
use serde::Serialize;

use npc_engine_core::AgentId;

use crate::SPRITE_SIZE;

#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize)]
pub struct Inventory(pub BTreeMap<AgentId, AgentInventory>);

#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Hash)]
pub struct AgentInventory {
    pub wood: isize,
    pub water: bool,
}

impl Inventory {
    pub fn draw(&self, ctx: &mut Context, assets: &BTreeMap<String, Image>) {
        let mut agents = self
            .0
            .iter()
            .map(|(k, v)| (*k, v.wood, v.water))
            .collect::<Vec<(AgentId, isize, bool)>>();

        agents.sort_by_key(|(k, ..)| *k);

        for (i, (agent, wood, water)) in agents.iter().enumerate() {
            let sprite_name = if agent.0 % 2 == 0 {
                "OrangeRight".to_owned()
            } else {
                "YellowRight".to_owned()
            };

            draw(
                ctx,
                assets.get(&sprite_name).unwrap(),
                ([0 as f32 * SPRITE_SIZE, i as f32 * SPRITE_SIZE], WHITE),
            )
            .unwrap();

            draw(
                ctx,
                &Text::new(format!(":{}, {}", wood, water)),
                ([
                    SPRITE_SIZE,
                    i as f32 * SPRITE_SIZE + (SPRITE_SIZE - DEFAULT_FONT_SCALE) / 2.,
                ],),
            )
            .unwrap();
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InventorySnapshot(pub BTreeMap<AgentId, AgentInventory>);

#[derive(Clone, Debug, Default, Hash, PartialEq, Eq)]
pub struct InventoryDiff(pub BTreeMap<AgentId, AgentInventory>);

impl InventoryDiff {
    pub fn diff_size(&self) -> usize {
        self.0.len() * mem::size_of::<(AgentId, AgentInventory)>()
    }
}
