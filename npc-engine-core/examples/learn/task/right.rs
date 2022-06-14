/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use npc_engine_core::{
    impl_task_boxed_methods, AgentId, StateDiffRef, StateDiffRefMut, Task, TaskDuration,
};
use npc_engine_utils::OptionDiffDomain;

use crate::domain::{DisplayAction, LearnDomain};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Right;
impl Task<LearnDomain> for Right {
    fn duration(
        &self,
        _tick: u64,
        _state_diff: StateDiffRef<LearnDomain>,
        _agent: AgentId,
    ) -> TaskDuration {
        1
    }

    fn execute(
        &self,
        _tick: u64,
        state_diff: StateDiffRefMut<LearnDomain>,
        _agent: AgentId,
    ) -> Option<Box<dyn Task<LearnDomain>>> {
        let state = LearnDomain::get_cur_state_mut(state_diff);
        debug_assert!((state.agent_pos as usize) < state.map.len() - 1);
        state.agent_pos += 1;
        None
    }

    fn is_valid(&self, _tick: u64, state_diff: StateDiffRef<LearnDomain>, _agent: AgentId) -> bool {
        let state = LearnDomain::get_cur_state(state_diff);
        (state.agent_pos as usize) < state.map.len() - 1
    }

    fn display_action(&self) -> DisplayAction {
        DisplayAction::Right
    }

    impl_task_boxed_methods!(LearnDomain);
}
