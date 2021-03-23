use std::fmt;
use std::hash::{Hash, Hasher};

use npc_engine_core::{AgentId, StateRef, StateRefMut, Task};

use crate::{config, Action, Direction, Lumberjacks, State, StateMut, Tile};

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
    fn weight(&self, _: StateRef<Lumberjacks>, _: AgentId) -> f32 {
        config().action_weights.plant
    }

    fn execute(
        &self,
        mut state: StateRefMut<Lumberjacks>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<Lumberjacks>>> {
        state.increment_time();

        if let Some((x, y)) = state.find_agent(agent) {
            let (x, y) = self.direction.apply(x, y);
            state.set_action(agent, Action::Plant(self.direction));

            match state.get_tile_ref_mut(x, y) {
                Some(tile @ Tile::Empty) => {
                    *tile = Tile::Tree(1);
                }
                _ => return None,
            }

            state.decrement_inventory(agent);

            None
        } else {
            unreachable!("Could not find agent on map!")
        }
    }

    fn valid(&self, state: StateRef<Lumberjacks>, agent: AgentId) -> bool {
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
