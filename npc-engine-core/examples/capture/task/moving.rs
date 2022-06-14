/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use npc_engine_core::{
    impl_task_boxed_methods, AgentId, StateDiffRef, StateDiffRefMut, Task, TaskDuration,
};
use npc_engine_utils::OptionDiffDomain;

use crate::{
    constants::MAP,
    domain::{CaptureDomain, DisplayAction},
    map::Location,
};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct StartMoving {
    pub to: Location,
}
impl Task<CaptureDomain> for StartMoving {
    fn duration(
        &self,
        _tick: u64,
        _state_diff: StateDiffRef<CaptureDomain>,
        _agent: AgentId,
    ) -> TaskDuration {
        // Start moving is instantaneous
        0
    }

    fn execute(
        &self,
        _tick: u64,
        state_diff: StateDiffRefMut<CaptureDomain>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<CaptureDomain>>> {
        let diff = CaptureDomain::get_cur_state_mut(state_diff);
        let agent_state = diff.agents.get_mut(&agent).unwrap();
        let to = self.to;
        agent_state.next_location = Some(to);
        // After starting, the agent must complete the move.
        let from = agent_state.cur_or_last_location;
        Some(Box::new(Moving { from, to }))
    }

    fn display_action(&self) -> DisplayAction {
        DisplayAction::StartMoving(self.to)
    }

    fn is_valid(
        &self,
        _tick: u64,
        state_diff: StateDiffRef<CaptureDomain>,
        agent: AgentId,
    ) -> bool {
        let state = CaptureDomain::get_cur_state(state_diff);
        state.agents.get(&agent).map_or(false, |agent_state| {
            let location = agent_state.cur_or_last_location;
            // We must be at a location (i.e. not moving).
            agent_state.next_location.is_none() &&
				// There must be a path to target.
				MAP.is_path(location, self.to)
        })
    }

    impl_task_boxed_methods!(CaptureDomain);
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Moving {
    from: Location,
    to: Location,
}
impl Task<CaptureDomain> for Moving {
    fn duration(
        &self,
        _tick: u64,
        _state_diff: StateDiffRef<CaptureDomain>,
        _agent: AgentId,
    ) -> TaskDuration {
        MAP.path_len(self.from, self.to).unwrap()
    }

    fn execute(
        &self,
        _tick: u64,
        state_diff: StateDiffRefMut<CaptureDomain>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<CaptureDomain>>> {
        let diff = CaptureDomain::get_cur_state_mut(state_diff);
        let agent_state = diff.agents.get_mut(&agent).unwrap();
        agent_state.cur_or_last_location = self.to;
        agent_state.next_location = None;
        // The agent has completed the move, it is now idle.
        None
    }

    fn display_action(&self) -> DisplayAction {
        DisplayAction::Moving(self.to)
    }

    fn is_valid(
        &self,
        _tick: u64,
        state_diff: StateDiffRef<CaptureDomain>,
        agent: AgentId,
    ) -> bool {
        let state = CaptureDomain::get_cur_state(state_diff);
        // This is a follow-up of StartMoving, so as the map is static, we assume
        // that as long as the agent exists, the task is valid.
        state.agents.get(&agent).is_some()
    }

    impl_task_boxed_methods!(CaptureDomain);
}
