
use std::hash::{Hash, Hasher};

use npc_engine_turn::{AgentId, Task, StateDiffRef, StateDiffRefMut, Domain, impl_task_boxed_methods};

use crate::{config, Action, Direction, Lumberjacks, Tile, WorldStateMut, WorldState};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Move {
    pub path: Vec<Direction>,
    pub x: usize,
    pub y: usize,
}

impl Task<Lumberjacks> for Move {
    fn weight(&self, _: StateDiffRef<Lumberjacks>, _: AgentId) -> f32 {
        config().action_weights.r#move
    }

    fn execute(
        &self,
        mut state_diff: StateDiffRefMut<Lumberjacks>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<Lumberjacks>>> {
        state_diff.increment_time();

        if let Some((x, y)) = state_diff.find_agent(agent) {
            let direction = self.path.first().unwrap();
            let (_x, _y) = direction.apply(x, y);
            state_diff.set_tile(x, y, Tile::Empty);
            state_diff.set_tile(_x, _y, Tile::Agent(agent));

            let path = self.path.iter().skip(1).copied().collect::<Vec<_>>();

            if !path.is_empty() {
                Some(Box::new(Move {
                    path,
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
        if let Some((mut x, mut y)) = state_diff.find_agent(agent) {
            self.path.iter().enumerate().all(|(idx, direction)| {
                let tmp = direction.apply(x, y);
                x = tmp.0;
                y = tmp.1;
                state_diff
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

    impl_task_boxed_methods!(Lumberjacks);
}
