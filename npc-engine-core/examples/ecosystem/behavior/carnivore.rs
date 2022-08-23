/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use npc_engine_core::{Behavior, Context, Task};
use npc_engine_utils::DIRECTIONS;

use crate::{
    domain::EcosystemDomain,
    state::{Access, AgentType},
    task::{eat_herbivore::EatHerbivore, jump::Jump},
};

use super::{animal::Animal, world::WORLD_AGENT_ID};

pub struct Carnivore;

impl Behavior<EcosystemDomain> for Carnivore {
    fn get_dependent_behaviors(&self) -> &'static [&'static dyn Behavior<EcosystemDomain>] {
        &[&Animal]
    }

    fn add_own_tasks(
        &self,
        ctx: Context<EcosystemDomain>,
        tasks: &mut Vec<Box<dyn Task<EcosystemDomain>>>,
    ) {
        for direction in DIRECTIONS {
            let jump_task = Jump(direction);
            if jump_task.is_valid(ctx) {
                tasks.push(Box::new(jump_task));
            }
            let eat_task = EatHerbivore(direction);
            if eat_task.is_valid(ctx) {
                tasks.push(Box::new(eat_task));
            }
        }
    }

    fn is_valid(&self, ctx: Context<EcosystemDomain>) -> bool {
        if ctx.agent == WORLD_AGENT_ID {
            return false;
        }
        ctx.state_diff
            .get_agent(ctx.agent)
            .filter(|agent_state| {
                // debug_assert!(agent_state.alive, "Behavior validity check called on a dead agent");
                agent_state.alive() && agent_state.ty == AgentType::Carnivore
            })
            .is_some()
    }
}
