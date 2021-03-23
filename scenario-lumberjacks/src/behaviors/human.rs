use std::fmt;

use npc_engine_core::{AgentId, Behavior, StateRef};

use crate::Lumberjacks;

pub struct Human;

impl fmt::Display for Human {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Human")
    }
}

impl Behavior<Lumberjacks> for Human {
    fn predicate(&self, _state: StateRef<Lumberjacks>, _agent: AgentId) -> bool {
        true
    }
}
