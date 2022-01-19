use std::{num::NonZeroU8};
use std::hash::{Hash, Hasher};

use npc_engine_common::{AgentId, Task, StateDiffRef, StateDiffRefMut, Domain, impl_task_boxed_methods};

use crate::{config, Action, Direction, Lumberjacks, Tile, DIRECTIONS, WorldStateMut, WorldState};

// SAFETY: this is safe as 1 is non-zero. This is actually a work-around the fact
// that Option::unwrap() is currently not const, but we need a constant in the match arm below.
// See the related Rust issue: https://github.com/rust-lang/rust/issues/67441
const NON_ZERO_U8_1: NonZeroU8 = unsafe { NonZeroU8::new_unchecked(1) };

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Chop {
    pub direction: Direction,
}

impl Task<Lumberjacks> for Chop {
    fn weight(&self, _: StateDiffRef<Lumberjacks>, _: AgentId) -> f32 {
        config().action_weights.chop
    }

    fn execute(
        &self,
        mut state_diff: StateDiffRefMut<Lumberjacks>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<Lumberjacks>>> {
        state_diff.increment_time();

        if let Some((x, y)) = state_diff.find_agent(agent) {
            let (x, y) = self.direction.apply(x, y);

            match state_diff.get_tile_ref_mut(x, y) {
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
                    if let Some(Tile::Agent(agent)) = state_diff.get_tile(x, y) {
                        state_diff.increment_inventory(agent);
                    }
                }
            } else {
                state_diff.increment_inventory(agent);
            }

            None
        } else {
            unreachable!("Could not find agent on map!")
        }
    }

    fn display_action(&self) -> <Lumberjacks as Domain>::DisplayAction {
        Action::Chop(self.direction)
    }

    fn is_valid(&self, state_diff: StateDiffRef<Lumberjacks>, agent: AgentId) -> bool {
        if let Some((x, y)) = state_diff.find_agent(agent) {
            let (x, y) = self.direction.apply(x, y);
            matches!(state_diff.get_tile(x, y), Some(Tile::Tree(_)))
        } else {
            unreachable!("Could not find agent on map!")
        }
    }

    impl_task_boxed_methods!(Lumberjacks);
}
