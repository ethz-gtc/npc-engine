use crate::{Lumberjacks, State, GlobalStateRef};
use npc_engine_turn::AgentId;

pub(crate) fn minimalist(state: GlobalStateRef<Lumberjacks>, _agent: AgentId) -> f32 {
    -(state.trees().len() as f32)
}
