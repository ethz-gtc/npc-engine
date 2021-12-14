use std::fmt;
use std::hash::{Hash, Hasher};

use npc_engine_turn::{AgentId, Task, StateDiffRef, StateDiffRefMut, Domain, impl_task_boxed_methods};

use crate::{config, Action, Lumberjacks, State, StateMut, GlobalStateRef, Tile, DIRECTIONS};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Refill;

impl fmt::Display for Refill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Refill")
    }
}

impl Task<Lumberjacks> for Refill {
    fn weight(&self, _: StateDiffRef<Lumberjacks>, _: AgentId) -> f32 {
        config().action_weights.refill
    }

    fn execute(
        &self,
        mut state_diff: StateDiffRefMut<Lumberjacks>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<Lumberjacks>>> {
        // FIXME: cleanup compat code
        state_diff.increment_time();

        state_diff.set_water(agent, true);
        None
    }

    fn display_action(&self) -> <Lumberjacks as Domain>::DisplayAction {
        Action::Refill
    }

    fn is_valid(&self, state_diff: StateDiffRef<Lumberjacks>, agent: AgentId) -> bool {
        // FIXME: cleanup compat code
        let state = GlobalStateRef::Snapshot(state_diff);
        if let Some((x, y)) = state.find_agent(agent) {
            !state.get_water(agent)
                && DIRECTIONS.iter().any(|direction| {
                    let (x, y) = direction.apply(x, y);
                    matches!(state.get_tile(x, y), Some(Tile::Well))
                })
        } else {
            unreachable!("Failed to find agent on map");
        }
    }

    impl_task_boxed_methods!(Lumberjacks);
}
