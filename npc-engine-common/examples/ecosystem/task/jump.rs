/* 
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

// use std::fmt::{self, Formatter};

use npc_engine_common::{Task, impl_task_boxed_methods, StateDiffRef, AgentId, TaskDuration, StateDiffRefMut};
use npc_engine_utils::Direction;

use crate::{domain::{EcosystemDomain, DisplayAction}, state::{Access, AccessMut}, map::DirConv};


#[derive(Hash, Clone, Eq, PartialEq, Debug)]
pub struct Jump(pub Direction);

impl Task<EcosystemDomain> for Jump {
    fn weight(&self, _tick: u64, _state_diff: StateDiffRef<EcosystemDomain>, _agent: AgentId) -> f32 {
        10.0
    }

    fn duration(&self, _tick: u64, _state_diff: StateDiffRef<EcosystemDomain>, _agent: AgentId) -> TaskDuration {
        0
    }

	fn execute(&self, _tick: u64, mut state_diff: StateDiffRefMut<EcosystemDomain>, agent: AgentId) -> Option<Box<dyn Task<EcosystemDomain>>> {
        let agent_state = state_diff.get_agent_mut(agent).unwrap();
		let passage_pos = DirConv::apply(self.0, agent_state.position);
        let target_pos = DirConv::apply(self.0, passage_pos);
        agent_state.position = target_pos;
        agent_state.food -= 1;
        None
    }

    fn is_valid(&self, _tick: u64, state_diff: StateDiffRef<EcosystemDomain>, agent: AgentId) -> bool {
        let agent_state = state_diff.get_agent(agent).unwrap();
        debug_assert!(agent_state.alive, "Task validity check called on a dead agent");
        if !agent_state.alive || agent_state.food <= 1 {
            return false;
        }

        let passage_pos = DirConv::apply(self.0, agent_state.position);
        let target_pos = DirConv::apply(self.0, passage_pos);
        state_diff.is_tile_passable(passage_pos) && state_diff.is_position_free(target_pos)
    }

	fn display_action(&self) -> DisplayAction {
        DisplayAction::Jump(self.0)
    }

	impl_task_boxed_methods!(EcosystemDomain);
}