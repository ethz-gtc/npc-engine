use std::{fmt, num::NonZeroU8};
use std::hash::{Hash, Hasher};

use npc_engine_turn::{AgentId, Task, StateDiffRef, StateDiffRefMut, Domain};

use crate::{config, Action, Direction, Lumberjacks, State, StateMut, GlobalStateRef, Tile};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Plant {
    pub direction: Direction,
}

impl fmt::Display for Plant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Plant({})", self.direction)
    }
}

impl Task<Lumberjacks> for Plant {
    fn weight(&self, _: StateDiffRef<Lumberjacks>, _: AgentId) -> f32 {
        config().action_weights.plant
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
            let (x, y) = self.direction.apply(x, y);

            match state_diff.get_tile_ref_mut(x, y) {
                Some(tile @ Tile::Empty) => {
                    *tile = Tile::Tree(NonZeroU8::new(1).unwrap());
                }
                _ => return None,
            }

            state_diff.decrement_inventory(agent);

            None
        } else {
            unreachable!("Could not find agent on map!")
        }
    }

    fn display_action(&self) -> <Lumberjacks as Domain>::DisplayAction {
        Action::Plant(self.direction)
    }

    fn is_valid(&self, state_diff: StateDiffRef<Lumberjacks>, agent: AgentId) -> bool {
        // FIXME: cleanup compat code
        let state = GlobalStateRef::Snapshot(state_diff);
        if let Some((x, y)) = state.find_agent(agent) {
            let (x, y) = self.direction.apply(x, y);
            matches!(state.get_tile(x, y), Some(Tile::Empty))
        } else {
            unreachable!("Could not find agent on map!")
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
