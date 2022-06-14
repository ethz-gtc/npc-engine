/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use npc_engine_core::{AgentId, Behavior, StateDiffRef, Task};
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
        tick: u64,
        state: StateDiffRef<EcosystemDomain>,
        agent: AgentId,
        tasks: &mut Vec<Box<dyn Task<EcosystemDomain>>>,
    ) {
        for direction in DIRECTIONS {
            let jump_task = Jump(direction);
            if jump_task.is_valid(tick, state, agent) {
                tasks.push(Box::new(jump_task));
            }
            let eat_task = EatHerbivore(direction);
            if eat_task.is_valid(tick, state, agent) {
                tasks.push(Box::new(eat_task));
            }
        }
    }

    fn is_valid(&self, _tick: u64, state: StateDiffRef<EcosystemDomain>, agent: AgentId) -> bool {
        if agent == WORLD_AGENT_ID {
            return false;
        }
        state
            .get_agent(agent)
            .filter(|agent_state| {
                // debug_assert!(agent_state.alive, "Behavior validity check called on a dead agent");
                agent_state.alive && agent_state.ty == AgentType::Carnivore
            })
            .is_some()
    }
}
