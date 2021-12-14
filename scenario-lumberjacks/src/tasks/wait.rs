use std::hash::{Hash, Hasher};

use npc_engine_turn::{AgentId, Task, StateDiffRef, StateDiffRefMut, Domain, impl_task_boxed_methods};

use crate::{config, Action, Lumberjacks, StateMut};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Wait;

impl Task<Lumberjacks> for Wait {
    fn weight(&self, _: StateDiffRef<Lumberjacks>, _: AgentId) -> f32 {
        config().action_weights.wait
    }

    fn execute(
        &self,
        mut state_diff: StateDiffRefMut<Lumberjacks>,
        _agent: AgentId,
    ) -> Option<Box<dyn Task<Lumberjacks>>> {
        state_diff.increment_time();

        None
    }

    fn display_action(&self) -> <Lumberjacks as Domain>::DisplayAction {
        Action::Wait
    }

    fn is_valid(&self, _: StateDiffRef<Lumberjacks>, _: AgentId) -> bool {
        true
    }

    impl_task_boxed_methods!(Lumberjacks);
}
