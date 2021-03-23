use std::fmt;
use std::hash::{Hash, Hasher};

use npc_engine_core::{AgentId, StateRef, StateRefMut, Task};

use crate::{config, Action, Lumberjacks, StateMut};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Wait;

impl fmt::Display for Wait {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Wait")
    }
}

impl Task<Lumberjacks> for Wait {
    fn weight(&self, _: StateRef<Lumberjacks>, _: AgentId) -> f32 {
        config().action_weights.wait
    }

    fn execute(
        &self,
        mut state: StateRefMut<Lumberjacks>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<Lumberjacks>>> {
        state.increment_time();

        state.set_action(agent, Action::Wait);
        None
    }

    fn valid(&self, _: StateRef<Lumberjacks>, _: AgentId) -> bool {
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
