use std::fmt;
use std::hash::{Hash, Hasher};

use npc_engine_turn::{AgentId, StateRef, StateRefMut, Task};

use crate::{config, Action, Lumberjacks, State, StateMut, Tile, DIRECTIONS};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Refill;

impl fmt::Display for Refill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Refill")
    }
}

impl Task<Lumberjacks> for Refill {
    fn weight(&self, _: StateRef<Lumberjacks>, _: AgentId) -> f32 {
        config().action_weights.refill
    }

    fn execute(
        &self,
        mut state: StateRefMut<Lumberjacks>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<Lumberjacks>>> {
        state.increment_time();

        state.set_action(agent, Action::Refill);
        state.set_water(agent, true);
        None
    }

    fn is_valid(&self, state: StateRef<Lumberjacks>, agent: AgentId) -> bool {
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
