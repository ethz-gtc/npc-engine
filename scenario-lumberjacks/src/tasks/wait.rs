use std::fmt;
use std::hash::{Hash, Hasher};

use npc_engine_turn::{AgentId, Task, SnapshotDiffRef, SnapshotDiffRefMut, Domain};

use crate::{config, Action, Lumberjacks, State, StateRef, StateRefMut, StateMut};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Wait;

impl fmt::Display for Wait {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Wait")
    }
}

impl Task<Lumberjacks> for Wait {
    fn weight(&self, _: SnapshotDiffRef<Lumberjacks>, _: AgentId) -> f32 {
        config().action_weights.wait
    }

    fn execute(
        &self,
        mut snapshot: SnapshotDiffRefMut<Lumberjacks>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<Lumberjacks>>> {
        // FIXME: cleanup compat code
        let mut state = StateRefMut::Snapshot(snapshot);
        state.increment_time();

        None
    }

    fn display_action(&self) -> <Lumberjacks as Domain>::DisplayAction {
        Action::Wait
    }

    fn is_valid(&self, _: SnapshotDiffRef<Lumberjacks>, _: AgentId) -> bool {
        true
    }

    fn box_clone(&self) -> Box<dyn Task<Lumberjacks>> {
        Box::new(self.clone())
    }

    fn box_hash(&self, mut state: &mut dyn Hasher) {
        self.hash(&mut state)
    }

    fn box_eq(&self, other: &Box<dyn Task<Lumberjacks>>) -> bool {
        if let Some(other) = other.downcast_ref::<Self>() {
            self.eq(other)
        } else {
            false
        }
    }
}
