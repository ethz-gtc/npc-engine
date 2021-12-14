use std::hash::{Hash, Hasher};

use npc_engine_turn::{AgentId, Task, StateDiffRef, StateDiffRefMut, Domain, impl_task_boxed_methods};

use crate::{config, Action, Direction, Lumberjacks, State, StateMut, Tile, DIRECTIONS};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Barrier {
    pub direction: Direction,
}

impl Task<Lumberjacks> for Barrier {
    fn weight(&self, _: StateDiffRef<Lumberjacks>, _: AgentId) -> f32 {
        config().action_weights.barrier
    }

    fn execute(
        &self,
        mut state_diff: StateDiffRefMut<Lumberjacks>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<Lumberjacks>>> {
        state_diff.increment_time();

        if let Some((x, y)) = state_diff.find_agent(agent) {
            let (x, y) = self.direction.apply(x, y);
            state_diff.set_tile(x, y, Tile::Barrier);
            state_diff.decrement_inventory(agent);

            None
        } else {
            unreachable!()
        }
    }

    fn display_action(&self) -> <Lumberjacks as Domain>::DisplayAction {
        Action::Barrier(self.direction)
    }

    fn is_valid(&self, state_diff: StateDiffRef<Lumberjacks>, agent: AgentId) -> bool {
        if let Some((x, y)) = state_diff.find_agent(agent) {
            let (x, y) = self.direction.apply(x, y);
            let empty = matches!(state_diff.get_tile(x, y), Some(Tile::Empty));
            let supported = DIRECTIONS
                .iter()
                .filter(|direction| {
                    let (x, y) = direction.apply(x, y);
                    state_diff
                        .get_tile(x, y)
                        .map(|tile| tile.is_support())
                        .unwrap_or(false)
                })
                .count()
                >= 1;

            empty && supported
        } else {
            unreachable!()
        }
    }

    impl_task_boxed_methods!(Lumberjacks);
}
