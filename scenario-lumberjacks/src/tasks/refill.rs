use std::hash::{Hash, Hasher};

use npc_engine_common::{AgentId, Task, StateDiffRef, StateDiffRefMut, Domain, impl_task_boxed_methods};

use crate::{config, Action, Lumberjacks, WorldState, WorldStateMut, Tile, DIRECTIONS};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Refill;

impl Task<Lumberjacks> for Refill {
    fn weight(&self, _: StateDiffRef<Lumberjacks>, _: AgentId) -> f32 {
        config().action_weights.refill
    }

    fn execute(
        &self,
        mut state_diff: StateDiffRefMut<Lumberjacks>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<Lumberjacks>>> {
        state_diff.increment_time();

        state_diff.set_water(agent, true);
        None
    }

    fn display_action(&self) -> <Lumberjacks as Domain>::DisplayAction {
        Action::Refill
    }

    fn is_valid(&self, state_diff: StateDiffRef<Lumberjacks>, agent: AgentId) -> bool {
        if let Some((x, y)) = state_diff.find_agent(agent) {
            !state_diff.get_water(agent)
                && DIRECTIONS.iter().any(|direction| {
                    let (x, y) = direction.apply(x, y);
                    matches!(state_diff.get_tile(x, y), Some(Tile::Well))
                })
        } else {
            unreachable!("Failed to find agent on map");
        }
    }

    impl_task_boxed_methods!(Lumberjacks);
}
