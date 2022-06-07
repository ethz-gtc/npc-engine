/* 
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich, see AUTHORS.txt for details
 */

use std::hash::Hash;

use npc_engine_common::{AgentId, Task, StateDiffRef, StateDiffRefMut, Domain, impl_task_boxed_methods, IdleTask, TaskDuration};
use npc_engine_utils::{Direction, DIRECTIONS};

use crate::{config, Action, Lumberjacks, WorldState, WorldStateMut, Tile, apply_direction};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Barrier {
    pub direction: Direction,
}

impl Task<Lumberjacks> for Barrier {
    fn weight(&self, _: u64, _: StateDiffRef<Lumberjacks>, _: AgentId) -> f32 {
        config().action_weights.barrier
    }

    fn duration(&self, _tick: u64, _state_diff: StateDiffRef<Lumberjacks>, _agent: AgentId) -> TaskDuration {
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
            state_diff.set_tile(x, y, Tile::Barrier);
            state_diff.decrement_inventory(agent);

            Some(Box::new(IdleTask))
        } else {
            unreachable!()
        }
    }

    fn display_action(&self) -> <Lumberjacks as Domain>::DisplayAction {
        Action::Barrier(self.direction)
    }

    fn is_valid(&self, _tick: u64,state_diff: StateDiffRef<Lumberjacks>, agent: AgentId) -> bool {
        if let Some((x, y)) = state_diff.find_agent(agent) {
            let (x, y) = apply_direction(self.direction, x, y);
            let empty = matches!(state_diff.get_tile(x, y), Some(Tile::Empty));
            let supported = DIRECTIONS
                .into_iter()
                .filter(|direction| {
                    let (x, y) = apply_direction(*direction, x, y);
                    state_diff
                        .get_tile(x, y)
                        .map(|tile| tile.is_support())
                        .unwrap_or(false)
                })
                .count()
                >= 1;

            empty && supported
        } else {
            unreachable!()
        }
    }

    impl_task_boxed_methods!(Lumberjacks);
}
