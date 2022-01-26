use std::hash::Hash;

use npc_engine_common::{AgentId, Task, StateDiffRef, StateDiffRefMut, Domain, impl_task_boxed_methods};

use crate::{config, Action, Lumberjacks, WorldStateMut};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Wait;

impl Task<Lumberjacks> for Wait {
    fn weight(&self, _: u64, _: StateDiffRef<Lumberjacks>, _: AgentId) -> f32 {
        config().action_weights.wait
    }

    fn execute(
        &self,
        _tick: u64,
        mut state_diff: StateDiffRefMut<Lumberjacks>,
        _agent: AgentId,
    ) -> Option<Box<dyn Task<Lumberjacks>>> {
        state_diff.increment_time();

        None
    }

    fn display_action(&self) -> <Lumberjacks as Domain>::DisplayAction {
        Action::Wait
    }

    fn is_valid(&self, _: u64,_: StateDiffRef<Lumberjacks>, _: AgentId) -> bool {
        true
    }

    impl_task_boxed_methods!(Lumberjacks);
}
