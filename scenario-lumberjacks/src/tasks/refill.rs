use std::fmt;
use std::hash::{Hash, Hasher};

use npc_engine_turn::{AgentId, Task, SnapshotDiffRef, SnapshotDiffRefMut, Domain};

use crate::{config, Action, Direction, Lumberjacks, State, StateMut, StateRef, StateRefMut, Tile, DIRECTIONS};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Refill;

impl fmt::Display for Refill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Refill")
    }
}

impl Task<Lumberjacks> for Refill {
    fn weight(&self, _: SnapshotDiffRef<Lumberjacks>, _: AgentId) -> f32 {
        config().action_weights.refill
    }

    fn execute(
        &self,
        mut snapshot: SnapshotDiffRefMut<Lumberjacks>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<Lumberjacks>>> {
        // FIXME: cleanup compat code
        let mut state = StateRefMut::Snapshot(snapshot);
        state.increment_time();

        state.set_water(agent, true);
        None
    }

    fn display_action(&self) -> <Lumberjacks as Domain>::DisplayAction {
        Action::Refill
    }

    fn is_valid(&self, snapshot: SnapshotDiffRef<Lumberjacks>, agent: AgentId) -> bool {
        // FIXME: cleanup compat code
        let state = StateRef::Snapshot(snapshot);
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
