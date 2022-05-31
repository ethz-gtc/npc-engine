use npc_engine_common::{Behavior, StateDiffRef, AgentId, Task, IdleTask};
use npc_engine_utils::DIRECTIONS;

use crate::{domain::EcosystemDomain, task::r#move::Move, state::Access};

pub struct Animal;

impl Behavior<EcosystemDomain> for Animal {
    fn add_own_tasks(&self, tick: u64, state: StateDiffRef<EcosystemDomain>, agent: AgentId, tasks: &mut Vec<Box<dyn Task<EcosystemDomain>>>) {
        for direction in DIRECTIONS {
            let task = Move(direction);
            if task.is_valid(tick, state, agent) {
                tasks.push(Box::new(task));
            }
        }
        tasks.push(Box::new(IdleTask));
    }

    fn is_valid(&self, _tick: u64, state: StateDiffRef<EcosystemDomain>, agent: AgentId) -> bool {
        state.get_agent(agent)
			.filter(|agent_state| {
				// debug_assert!(agent_state.alive, "Behavior validity check called on a dead agent");
				agent_state.alive
			})
			.is_some()
    }
}
