use std::fmt;
use std::hash::{Hash, Hasher};

use npc_engine_turn::{AgentId, Task, StateDiffRef, StateDiffRefMut, Domain};

use crate::{config, Action, Direction, Lumberjacks, GlobalStateRef, GlobalStateRefMut, Tile, StateMut, State};

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
    fn weight(&self, _: StateDiffRef<Lumberjacks>, _: AgentId) -> f32 {
        config().action_weights.r#move
    }

    fn execute(
        &self,
        state_diff: StateDiffRefMut<Lumberjacks>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<Lumberjacks>>> {
        // FIXME: cleanup compat code
        let mut state = GlobalStateRefMut::Snapshot(state_diff);
        state.increment_time();

        if let Some((x, y)) = state.find_agent(agent) {
            let direction = self.path.first().unwrap();
            let (_x, _y) = direction.apply(x, y);
            state.set_tile(x, y, Tile::Empty);
            state.set_tile(_x, _y, Tile::Agent(agent));

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

    fn display_action(&self) -> <Lumberjacks as Domain>::DisplayAction {
        Action::Walk(*self.path.first().unwrap())
    }

    fn is_valid(&self, state_diff: StateDiffRef<Lumberjacks>, agent: AgentId) -> bool {
        // FIXME: cleanup compat code
        let state = GlobalStateRef::Snapshot(state_diff);
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
