/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use npc_engine_core::{Behavior, Context, IdleTask, Task};
use npc_engine_utils::DIRECTIONS;

use crate::{domain::EcosystemDomain, state::Access, task::r#move::Move};

pub struct Animal;

impl Behavior<EcosystemDomain> for Animal {
    fn add_own_tasks(
        &self,
        ctx: Context<EcosystemDomain>,
        tasks: &mut Vec<Box<dyn Task<EcosystemDomain>>>,
    ) {
        for direction in DIRECTIONS {
            let task = Move(direction);
            if task.is_valid(ctx) {
                tasks.push(Box::new(task));
            }
        }
        tasks.push(Box::new(IdleTask));
    }

    fn is_valid(&self, ctx: Context<EcosystemDomain>) -> bool {
        ctx.state_diff
            .get_agent(ctx.agent)
            .filter(|agent_state| {
                // debug_assert!(agent_state.alive, "Behavior validity check called on a dead agent");
                agent_state.alive()
            })
            .is_some()
    }
}
