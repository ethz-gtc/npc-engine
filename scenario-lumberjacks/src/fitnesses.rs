use crate::{Lumberjacks, State};
use npc_engine_turn::{AgentId, StateDiffRef};

pub(crate) fn minimalist(state: StateDiffRef<Lumberjacks>, _agent: AgentId) -> f32 {
    -(state.trees().len() as f32)
}
