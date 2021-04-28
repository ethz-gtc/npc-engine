use std::{fmt, num::NonZeroU8};
use std::hash::{Hash, Hasher};

use npc_engine_core::{AgentId, StateRef, StateRefMut, Task};

use crate::{config, Action, Direction, Lumberjacks, State, StateMut, Tile, DIRECTIONS};

// SAFETY: this is safe as 1 is non-zero. This is actually a work-around the fact
// that Option::unwrap() is currently not const, but we need a constant in the match arm below.
// See the related Rust issue: https://github.com/rust-lang/rust/issues/67441
const NON_ZERO_U8_1: NonZeroU8 = unsafe { NonZeroU8::new_unchecked(1) };

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Chop {
    pub direction: Direction,
}

impl fmt::Display for Chop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Chop({})", self.direction)
    }
}

impl Task<Lumberjacks> for Chop {
    fn weight(&self, _: StateRef<Lumberjacks>, _: AgentId) -> f32 {
        config().action_weights.chop
    }

    fn execute(
        &self,
        mut state: StateRefMut<Lumberjacks>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<Lumberjacks>>> {
        state.increment_time();

        if let Some((x, y)) = state.find_agent(agent) {
            let (x, y) = self.direction.apply(x, y);
            state.set_action(agent, Action::Chop(self.direction));

            match state.get_tile_ref_mut(x, y) {
                Some(tile @ Tile::Tree(NON_ZERO_U8_1)) => {
                    *tile = Tile::Empty;
                }
                Some(Tile::Tree(height)) => {
                    *height = NonZeroU8::new(height.get() - 1).unwrap();
                }
                _ => return None,
            }

            if config().features.teamwork {
                for direction in DIRECTIONS {
                    let (x, y) = direction.apply(x, y);
                    if let Some(Tile::Agent(agent)) = state.get_tile(x, y) {
                        state.increment_inventory(agent);
                    }
                }
            } else {
                state.increment_inventory(agent);
            }

            None
        } else {
            unreachable!("Could not find agent on map!")
        }
    }

    fn valid(&self, state: StateRef<Lumberjacks>, agent: AgentId) -> bool {
        if let Some((x, y)) = state.find_agent(agent) {
            let (x, y) = self.direction.apply(x, y);
            matches!(state.get_tile(x, y), Some(Tile::Tree(_)))
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
