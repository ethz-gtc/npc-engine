use std::hash::Hash;

use npc_engine_common::{AgentId, Task, StateDiffRef, StateDiffRefMut, Domain, impl_task_boxed_methods, TaskDuration, IdleTask};

use crate::{config, Action, Direction, Lumberjacks, WorldState, WorldStateMut, Tile};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Water {
    pub direction: Direction,
}

impl Task<Lumberjacks> for Water {
    fn weight(&self, _: u64, _: StateDiffRef<Lumberjacks>, _: AgentId) -> f32 {
        config().action_weights.water
    }

    fn duration(&self, _tick: u64, _state_diff: StateDiffRef<Lumberjacks>, _agent: AgentId) -> TaskDuration {
        0
    }

    fn execute(
        &self,
        _tick: u64,
        mut state_diff: StateDiffRefMut<Lumberjacks>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<Lumberjacks>>> {
        state_diff.increment_time();

        if let Some((x, y)) = state_diff.find_agent(agent) {
            state_diff.set_water(agent, false);

            let (x, y) = self.direction.apply(x, y);
            if let Some(Tile::Tree(height)) = state_diff.get_tile_ref_mut(x, y) {
                *height = config().map.tree_height;
            }

            Some(Box::new(IdleTask))
        } else {
            unreachable!("Failed to find agent on map");
        }
    }

    fn display_action(&self) -> <Lumberjacks as Domain>::DisplayAction {
        Action::Water(self.direction)
    }

    fn is_valid(&self, _tick: u64, state_diff: StateDiffRef<Lumberjacks>, agent: AgentId) -> bool {
        state_diff.get_water(agent)
            && if let Some((x, y)) = state_diff.find_agent(agent) {
                let (x, y) = self.direction.apply(x, y);
                matches!(state_diff.get_tile(x, y), Some(Tile::Tree(_)))
            } else {
                unreachable!("Failed to find agent on map");
            }
    }

    impl_task_boxed_methods!(Lumberjacks);
}
