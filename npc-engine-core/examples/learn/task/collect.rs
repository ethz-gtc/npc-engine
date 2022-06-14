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
pub struct Collect;
impl Task<LearnDomain> for Collect {
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
        debug_assert!(state.map[state.agent_pos as usize] > 0);
        state.map[state.agent_pos as usize] -= 1;
        state.wood_count += 1;
        None
    }

    fn is_valid(&self, _tick: u64, state_diff: StateDiffRef<LearnDomain>, _agent: AgentId) -> bool {
        let state = LearnDomain::get_cur_state(state_diff);
        state.map[state.agent_pos as usize] > 0
    }

    fn display_action(&self) -> DisplayAction {
        DisplayAction::Collect
    }

    impl_task_boxed_methods!(LearnDomain);
}
