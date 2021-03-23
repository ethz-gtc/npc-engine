use std::fmt;
use std::hash::{Hash, Hasher};

use npc_engine_core::{AgentId, StateRef, StateRefMut, Task};

use crate::{config, Action, Direction, Lumberjacks, State, StateMut, Tile, DIRECTIONS};

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
                Some(tile @ Tile::Tree(1)) => {
                    *tile = Tile::Empty;
                }
                Some(Tile::Tree(height)) => {
                    *height -= 1;
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
