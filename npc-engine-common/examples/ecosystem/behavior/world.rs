use npc_engine_common::{AgentId, Behavior, StateDiffRef, Task};

use crate::{domain::EcosystemDomain, task::world::WorldStep};

pub const WORLD_AGENT_ID: AgentId = AgentId(u32::MAX);

pub struct WorldBehavior;
impl Behavior<EcosystemDomain> for WorldBehavior {
	fn add_own_tasks(
		&self,
		_tick: u64,
		_state_diff: StateDiffRef<EcosystemDomain>,
		_agent: AgentId,
		tasks: &mut Vec<Box<dyn Task<EcosystemDomain>>>,
	) {
		tasks.push(Box::new(WorldStep));
	}

	fn is_valid(&self, _tick: u64, _state_diff: StateDiffRef<EcosystemDomain>, agent: AgentId) -> bool {
		agent == WORLD_AGENT_ID
	}
}