/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::fmt;

use npc_engine_core::{
    impl_task_boxed_methods, Behavior, Context, ContextMut, IdleTask, Task, TaskDuration,
};

use crate::{
    board::{Board, Cell, CellArray2D, CellCoord, C_RANGE},
    domain::{DisplayAction, TicTacToe},
    player::Player,
};

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct Move {
    pub x: CellCoord,
    pub y: CellCoord,
}
impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.x, self.y)
    }
}

impl Task<TicTacToe> for Move {
    fn duration(&self, _ctx: Context<TicTacToe>) -> TaskDuration {
        // Moves affect the board instantly
        0
    }

    fn execute(&self, ctx: ContextMut<TicTacToe>) -> Option<Box<dyn Task<TicTacToe>>> {
        let diff = if let Some(diff) = ctx.state_diff.diff {
            diff
        } else {
            *ctx.state_diff.diff = Some(0);
            &mut *ctx.state_diff.diff.as_mut().unwrap()
        };
        diff.set(self.x, self.y, Cell::Player(Player::from_agent(ctx.agent)));
        assert!(ctx.state_diff.diff.is_some());
        // After every move, one has to wait one's next turn
        Some(Box::new(IdleTask))
    }

    fn display_action(&self) -> DisplayAction {
        DisplayAction(Some(self.clone()))
    }

    fn is_valid(&self, ctx: Context<TicTacToe>) -> bool {
        let state = *ctx.state_diff.initial_state | ctx.state_diff.diff.unwrap_or(0);
        state.winner().is_none() && state.get(self.x, self.y) == Cell::Empty
    }

    impl_task_boxed_methods!(TicTacToe);
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &'_ mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Move")
            .field("x", &self.x.get())
            .field("y", &self.y.get())
            .finish()
    }
}

pub struct MoveBehavior;
impl Behavior<TicTacToe> for MoveBehavior {
    fn add_own_tasks(&self, ctx: Context<TicTacToe>, tasks: &mut Vec<Box<dyn Task<TicTacToe>>>) {
        // if the game is already ended, no move are valid
        let board = *ctx.state_diff.initial_state | ctx.state_diff.diff.unwrap_or(0);
        if board.winner().is_some() {
            return;
        }
        for x in C_RANGE {
            for y in C_RANGE {
                let task = Move { x, y };
                if task.is_valid(ctx) {
                    tasks.push(Box::new(task));
                }
            }
        }
    }

    fn is_valid(&self, _ctx: Context<TicTacToe>) -> bool {
        true
    }
}
