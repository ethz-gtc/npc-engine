use std::fmt;

use npc_engine_turn::{AgentId, Behavior, StateDiffRef};

use crate::Lumberjacks;

pub struct Human;

impl fmt::Display for Human {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Human")
    }
}

impl Behavior<Lumberjacks> for Human {
    fn is_valid(&self, _state: StateDiffRef<Lumberjacks>, _agent: AgentId) -> bool {
        true
    }
}
