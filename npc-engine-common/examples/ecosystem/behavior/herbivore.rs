use npc_engine_common::{Behavior, StateDiffRef, AgentId, Task};

use crate::{domain::EcosystemDomain, state::{Access, AgentType}, task::eat_grass::EatGrass};

use super::{animal::Animal, world::WORLD_AGENT_ID};

pub struct Herbivore;

impl Behavior<EcosystemDomain> for Herbivore {
	fn get_dependent_behaviors(&self) -> &'static [&'static dyn Behavior<EcosystemDomain>] {
		&[&Animal]
	}

    fn add_own_tasks(&self, tick: u64, state: StateDiffRef<EcosystemDomain>, agent: AgentId, tasks: &mut Vec<Box<dyn Task<EcosystemDomain>>>) {
        let eat_task = EatGrass;
        if eat_task.is_valid(tick, state, agent) {
            tasks.push(Box::new(eat_task));
        }
    }

    fn is_valid(&self, _tick: u64, state: StateDiffRef<EcosystemDomain>, agent: AgentId) -> bool {
		if agent == WORLD_AGENT_ID {
			return false;
		}
        state.get_agent(agent)
			.filter(|agent_state| {
				// debug_assert!(agent_state.alive, "Behavior validity check called on a dead agent");
				agent_state.alive && agent_state.ty == AgentType::Herbivore
			})
			.is_some()
    }
}
