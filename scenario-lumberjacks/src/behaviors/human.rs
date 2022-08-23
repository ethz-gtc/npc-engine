/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::fmt;

use npc_engine_core::{Behavior, Context};

use crate::Lumberjacks;

pub struct Human;

impl fmt::Display for Human {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Human")
    }
}

impl Behavior<Lumberjacks> for Human {
    fn is_valid(&self, _ctx: Context<Lumberjacks>) -> bool {
        true
    }

    fn add_own_tasks(
        &self,
        _ctx: Context<Lumberjacks>,
        _tasks: &mut Vec<Box<dyn npc_engine_core::Task<Lumberjacks>>>,
    ) {
    }
}
