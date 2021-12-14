use std::fmt;
use std::hash::{Hash, Hasher};

use npc_engine_turn::{AgentId, Task, StateDiffRef, StateDiffRefMut, Domain};

use crate::{config, Action, Direction, Lumberjacks, State, StateMut, GlobalStateRef, Tile};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Water {
    pub direction: Direction,
}

impl fmt::Display for Water {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Water({})", self.direction)
    }
}

impl Task<Lumberjacks> for Water {
    fn weight(&self, _: StateDiffRef<Lumberjacks>, _: AgentId) -> f32 {
        config().action_weights.water
    }

    fn execute(
        &self,
        mut state_diff: StateDiffRefMut<Lumberjacks>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<Lumberjacks>>> {
        // FIXME: cleanup compat code
        let state = GlobalStateRef::Snapshot(*state_diff);
        state_diff.increment_time();

        if let Some((x, y)) = state.find_agent(agent) {
            state_diff.set_water(agent, false);

            let (x, y) = self.direction.apply(x, y);
            if let Some(Tile::Tree(height)) = state_diff.get_tile_ref_mut(x, y) {
                *height = config().map.tree_height;
            }

            None
        } else {
            unreachable!("Failed to find agent on map");
        }
    }

    fn display_action(&self) -> <Lumberjacks as Domain>::DisplayAction {
        Action::Water(self.direction)
    }

    fn is_valid(&self, state_diff: StateDiffRef<Lumberjacks>, agent: AgentId) -> bool {
        // FIXME: cleanup compat code
        let state = GlobalStateRef::Snapshot(state_diff);
        state.get_water(agent)
            && if let Some((x, y)) = state.find_agent(agent) {
                let (x, y) = self.direction.apply(x, y);
                matches!(state.get_tile(x, y), Some(Tile::Tree(_)))
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
