// use std::fmt::{self, Formatter};

use npc_engine_common::{Task, impl_task_boxed_methods, StateDiffRef, AgentId, TaskDuration, StateDiffRefMut};
use npc_engine_utils::Direction;

use crate::{domain::{EcosystemDomain, DisplayAction}, state::{Access, AccessMut}, map::DirConv};


#[derive(Hash, Clone, Eq, PartialEq, Debug)]
pub struct Move(pub Direction);

impl Task<EcosystemDomain> for Move {
    fn weight(&self, _tick: u64, _state_diff: StateDiffRef<EcosystemDomain>, _agent: AgentId) -> f32 {
        5.0
    }

    fn duration(&self, _tick: u64, _state_diff: StateDiffRef<EcosystemDomain>, _agent: AgentId) -> TaskDuration {
        0
    }

	fn execute(&self, _tick: u64, mut state_diff: StateDiffRefMut<EcosystemDomain>, agent: AgentId) -> Option<Box<dyn Task<EcosystemDomain>>> {
        let agent_state = state_diff.get_agent(agent).unwrap();
		let target_pos = DirConv::apply(self.0, agent_state.position);
		*state_diff.get_agent_pos_mut(agent).unwrap() = target_pos;
        None
    }

    fn is_valid(&self, _tick: u64, state_diff: StateDiffRef<EcosystemDomain>, agent: AgentId) -> bool {
        let agent_state = state_diff.get_agent(agent).unwrap();
        debug_assert!(agent_state.alive, "Task validity check called on a dead agent");
        if !agent_state.alive {
            return false;
        }

        let agent_pos = agent_state.position;
		let target_pos = DirConv::apply(self.0, agent_pos);
		if let Some(_agent) = state_diff.get_agent_at(target_pos) {
			return false;
		}

        state_diff.get_tile(target_pos)
			.map(|tile| tile.is_passable())
			.unwrap_or(false)
    }

	fn display_action(&self) -> DisplayAction {
        DisplayAction::Move(self.0)
    }

	impl_task_boxed_methods!(EcosystemDomain);
}