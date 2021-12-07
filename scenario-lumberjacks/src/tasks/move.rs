use std::fmt;
use std::hash::{Hash, Hasher};

use npc_engine_turn::{AgentId, StateRef, StateRefMut, Task};

use crate::{config, Action, Direction, Lumberjacks, State, StateMut, Tile};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Move {
    pub path: Vec<Direction>,
    pub x: usize,
    pub y: usize,
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Move({}, {})", self.x, self.y)
    }
}

impl Task<Lumberjacks> for Move {
    fn weight(&self, _: StateRef<Lumberjacks>, _: AgentId) -> f32 {
        config().action_weights.r#move
    }

    fn execute(
        &self,
        mut state: StateRefMut<Lumberjacks>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<Lumberjacks>>> {
        state.increment_time();

        if let Some((x, y)) = state.find_agent(agent) {
            let direction = self.path.first().unwrap();
            let (_x, _y) = direction.apply(x, y);
            state.set_tile(x, y, Tile::Empty);
            state.set_tile(_x, _y, Tile::Agent(agent));
            state.set_action(agent, Action::Walk(*direction));

            let path = self.path.iter().skip(1).copied().collect::<Vec<_>>();

            if !path.is_empty() {
                Some(Box::new(Move {
                    path: self.path.iter().skip(1).copied().collect(),
                    x: self.x,
                    y: self.y,
                }))
            } else {
                None
            }
        } else {
            unreachable!()
        }
    }

    fn is_valid(&self, state: StateRef<Lumberjacks>, agent: AgentId) -> bool {
        if let Some((mut x, mut y)) = state.find_agent(agent) {
            self.path.iter().enumerate().all(|(idx, direction)| {
                let tmp = direction.apply(x, y);
                x = tmp.0;
                y = tmp.1;
                state
                    .get_tile(x, y)
                    .map(|tile| {
                        if idx == 0 {
                            tile.is_walkable()
                        } else {
                            tile.is_pathfindable()
                        }
                    })
                    .unwrap_or(false)
            })
        } else {
            unreachable!()
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
