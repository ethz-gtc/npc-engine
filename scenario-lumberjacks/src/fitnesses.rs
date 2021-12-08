use crate::{Lumberjacks, State, StateRef};
use npc_engine_turn::AgentId;

pub(crate) fn minimalist(state: StateRef<Lumberjacks>, _agent: AgentId) -> f32 {
    -(state.trees().len() as f32)
}
