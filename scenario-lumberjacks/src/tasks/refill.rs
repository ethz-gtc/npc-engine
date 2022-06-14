/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::hash::Hash;

use npc_engine_core::{
    impl_task_boxed_methods, AgentId, Domain, IdleTask, StateDiffRef, StateDiffRefMut, Task,
    TaskDuration,
};
use npc_engine_utils::DIRECTIONS;

use crate::{apply_direction, config, Action, Lumberjacks, Tile, WorldState, WorldStateMut};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Refill;

impl Task<Lumberjacks> for Refill {
    fn weight(&self, _: u64, _: StateDiffRef<Lumberjacks>, _: AgentId) -> f32 {
        config().action_weights.refill
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

        state_diff.set_water(agent, true);
        Some(Box::new(IdleTask))
    }

    fn display_action(&self) -> <Lumberjacks as Domain>::DisplayAction {
        Action::Refill
    }

    fn is_valid(&self, _tick: u64, state_diff: StateDiffRef<Lumberjacks>, agent: AgentId) -> bool {
        if let Some((x, y)) = state_diff.find_agent(agent) {
            !state_diff.get_water(agent)
                && DIRECTIONS.iter().any(|direction| {
                    let (x, y) = apply_direction(*direction, x, y);
                    matches!(state_diff.get_tile(x, y), Some(Tile::Well))
                })
        } else {
            unreachable!("Failed to find agent on map");
        }
    }

    impl_task_boxed_methods!(Lumberjacks);
}
