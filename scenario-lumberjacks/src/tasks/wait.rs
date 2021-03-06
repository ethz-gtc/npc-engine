/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::hash::Hash;

use npc_engine_core::{
    impl_task_boxed_methods, AgentId, Domain, IdleTask, StateDiffRef, StateDiffRefMut, Task,
    TaskDuration,
};

use crate::{config, Action, Lumberjacks, WorldStateMut};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Wait;

impl Task<Lumberjacks> for Wait {
    fn weight(&self, _: u64, _: StateDiffRef<Lumberjacks>, _: AgentId) -> f32 {
        config().action_weights.wait
    }

    fn duration(
        &self,
        _tick: u64,
        _state_diff: StateDiffRef<Lumberjacks>,
        _agent: AgentId,
    ) -> TaskDuration {
        0
    }

    fn execute(
        &self,
        _tick: u64,
        mut state_diff: StateDiffRefMut<Lumberjacks>,
        _agent: AgentId,
    ) -> Option<Box<dyn Task<Lumberjacks>>> {
        state_diff.increment_time();

        Some(Box::new(IdleTask))
    }

    fn display_action(&self) -> <Lumberjacks as Domain>::DisplayAction {
        Action::Wait
    }

    fn is_valid(&self, _: u64, _: StateDiffRef<Lumberjacks>, _: AgentId) -> bool {
        true
    }

    impl_task_boxed_methods!(Lumberjacks);
}
