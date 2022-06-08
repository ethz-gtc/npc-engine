/* 
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::hash::Hash;

use npc_engine_common::{AgentId, Task, StateDiffRef, StateDiffRefMut, Domain, impl_task_boxed_methods, IdleTask, TaskDuration};
use npc_engine_utils::Direction;

use crate::{config, Action, Lumberjacks, Tile, WorldStateMut, WorldState, apply_direction};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Move {
    pub path: Vec<Direction>,
    pub x: usize,
    pub y: usize,
}

impl Task<Lumberjacks> for Move {
    fn weight(&self, _: u64, _: StateDiffRef<Lumberjacks>, _: AgentId) -> f32 {
        config().action_weights.r#move
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
            let direction = self.path.first().unwrap();
            let (_x, _y) = apply_direction(*direction, x, y);
            state_diff.set_tile(x, y, Tile::Empty);
            state_diff.set_tile(_x, _y, Tile::Agent(agent));

            let mut path = self.path.iter().skip(1).copied();

            if path.next().is_some() {
                panic!("Objectives are currently disabled in Lumberjack, so path do not make sense"); // objectives are disabled
                /*Some(Box::new(Move {
                    path,
                    x: self.x,
                    y: self.y,
                }))*/
            } else {
                Some(Box::new(IdleTask))
            }
        } else {
            unreachable!()
        }
    }

    fn display_action(&self) -> <Lumberjacks as Domain>::DisplayAction {
        Action::Walk(*self.path.first().unwrap())
    }

    fn is_valid(&self, _tick: u64,state_diff: StateDiffRef<Lumberjacks>, agent: AgentId) -> bool {
        if let Some((mut x, mut y)) = state_diff.find_agent(agent) {
            self.path.iter().enumerate().all(|(idx, direction)| {
                let tmp = apply_direction(*direction, x, y);
                x = tmp.0;
                y = tmp.1;
                state_diff
                    .get_tile(x, y)
                    .map(|tile| {
                        if idx == 0 {
                            tile.is_walkable()
                        } else {
                            tile.is_pathfindable()
                        }
                    })
                    .unwrap_or(false)
            })
        } else {
            unreachable!()
        }
    }

    impl_task_boxed_methods!(Lumberjacks);
}
