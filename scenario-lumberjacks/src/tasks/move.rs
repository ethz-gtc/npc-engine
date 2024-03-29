/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::hash::Hash;

use npc_engine_core::{
    impl_task_boxed_methods, Context, ContextMut, Domain, IdleTask, Task, TaskDuration,
};
use npc_engine_utils::Direction;

use crate::{apply_direction, config, Action, Lumberjacks, Tile, WorldState, WorldStateMut};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Move {
    pub path: Vec<Direction>,
    pub x: usize,
    pub y: usize,
}

impl Task<Lumberjacks> for Move {
    fn weight(&self, _ctx: Context<Lumberjacks>) -> f32 {
        config().action_weights.r#move
    }

    fn duration(&self, _ctx: Context<Lumberjacks>) -> TaskDuration {
        0
    }

    fn execute(&self, ctx: ContextMut<Lumberjacks>) -> Option<Box<dyn Task<Lumberjacks>>> {
        let ContextMut {
            mut state_diff,
            agent,
            ..
        } = ctx;
        state_diff.increment_time();

        if let Some((x, y)) = state_diff.find_agent(agent) {
            let direction = self.path.first().unwrap();
            let (_x, _y) = apply_direction(*direction, x, y);
            state_diff.set_tile(x, y, Tile::Empty);
            state_diff.set_tile(_x, _y, Tile::Agent(agent));

            let mut path = self.path.iter().skip(1).copied();

            if path.next().is_some() {
                panic!(
                    "Objectives are currently disabled in Lumberjack, so path do not make sense"
                ); // objectives are disabled
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

    fn is_valid(&self, ctx: Context<Lumberjacks>) -> bool {
        let Context {
            state_diff, agent, ..
        } = ctx;
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
