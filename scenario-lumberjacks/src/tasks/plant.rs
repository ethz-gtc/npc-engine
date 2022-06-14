/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::hash::Hash;
use std::num::NonZeroU8;

use npc_engine_core::{
    impl_task_boxed_methods, AgentId, Domain, IdleTask, StateDiffRef, StateDiffRefMut, Task,
    TaskDuration,
};
use npc_engine_utils::Direction;

use crate::{apply_direction, config, Action, Lumberjacks, Tile, WorldState, WorldStateMut};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Plant {
    pub direction: Direction,
}

impl Task<Lumberjacks> for Plant {
    fn weight(&self, _: u64, _: StateDiffRef<Lumberjacks>, _: AgentId) -> f32 {
        config().action_weights.plant
    }

    fn duration(
        &self,
        _tick: u64,
        _state_diff: StateDiffRef<Lumberjacks>,
        _agent: AgentId,
    ) -> TaskDuration {
        0
    }

    fn execute(
        &self,
        _tick: u64,
        mut state_diff: StateDiffRefMut<Lumberjacks>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<Lumberjacks>>> {
        state_diff.increment_time();

        if let Some((x, y)) = state_diff.find_agent(agent) {
            let (x, y) = apply_direction(self.direction, x, y);

            match state_diff.get_tile_ref_mut(x, y) {
                Some(tile @ Tile::Empty) => {
                    *tile = Tile::Tree(NonZeroU8::new(1).unwrap());
                }
                _ => return Some(Box::new(IdleTask)),
            }

            state_diff.decrement_inventory(agent);

            Some(Box::new(IdleTask))
        } else {
            unreachable!("Could not find agent on map!")
        }
    }

    fn display_action(&self) -> <Lumberjacks as Domain>::DisplayAction {
        Action::Plant(self.direction)
    }

    fn is_valid(&self, _tick: u64, state_diff: StateDiffRef<Lumberjacks>, agent: AgentId) -> bool {
        if let Some((x, y)) = state_diff.find_agent(agent) {
            let (x, y) = apply_direction(self.direction, x, y);
            matches!(state_diff.get_tile(x, y), Some(Tile::Empty))
        } else {
            unreachable!("Could not find agent on map!")
        }
    }

    impl_task_boxed_methods!(Lumberjacks);
}
