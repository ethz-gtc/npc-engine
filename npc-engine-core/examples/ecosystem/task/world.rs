/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use npc_engine_core::{
    impl_task_boxed_methods, AgentId, StateDiffRef, StateDiffRefMut, Task, TaskDuration,
};
use npc_engine_utils::{Coord2D, DIRECTIONS};

use crate::{
    constants::{
        WORLD_GRASS_EXPAND_CYCLE_DURATION, WORLD_GRASS_GROW_CYCLE_DURATION, WORLD_TASK_DURATION,
    },
    domain::{DisplayAction, EcosystemDomain},
    map::{DirConv, Tile},
    state::{Access, AccessMut, AgentState},
};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct WorldStep;

impl Task<EcosystemDomain> for WorldStep {
    fn duration(
        &self,
        _tick: u64,
        _state_diff: StateDiffRef<EcosystemDomain>,
        _agent: AgentId,
    ) -> TaskDuration {
        WORLD_TASK_DURATION
    }

    fn execute(
        &self,
        tick: u64,
        mut state_diff: StateDiffRefMut<EcosystemDomain>,
        _agent: AgentId,
    ) -> Option<Box<dyn Task<EcosystemDomain>>> {
        // 1. Consume food, every time, and make kids, if it is the right time
        for agent in state_diff.list_agents() {
            // Consume food
            let state = state_diff.get_agent_mut(agent).unwrap();
            if state.food > 0 {
                state.food -= 1;
                if state.food == 0 {
                    state.kill(tick)
                }
            }
            // Make baby
            // Note: It is important that only a single agent creates new agent, otherwise
            // there is a risk of collision in AgentId. In that case, either an atomic could be used,
            // but the agent id space will be consumed fast, or some form of agent id renaming will
            // need to be implemented.
            let baby_season = tick % state.ty.reproduction_cycle_duration() < WORLD_TASK_DURATION;
            if baby_season && state.food >= state.ty.min_food_for_reproduction() {
                let parent_position = state.position;
                for direction in DIRECTIONS {
                    let baby_position = DirConv::apply(direction, parent_position);
                    if state_diff.is_position_free(baby_position) {
                        // modify parent
                        let parent_state = state_diff.get_agent_mut(agent).unwrap();
                        let food = parent_state.ty.food_given_to_baby();
                        parent_state.food -= food;
                        let ty = parent_state.ty;
                        // create child
                        state_diff.new_agent(AgentState {
                            ty,
                            birth_date: tick,
                            position: baby_position,
                            food,
                            dead_tick: None,
                        });
                        break;
                    }
                }
            }
        }
        // 2. Grow grass, if it is time (1/WORLD_GRASS_REGROW_PERIOD of times)
        let width = state_diff.map_width();
        let height = state_diff.map_height();
        for y in 0..height {
            for x in 0..width {
                let handle_tick = tick
                    .wrapping_add((x as u64) * 2797)
                    .wrapping_add((y as u64) * 3637);
                let position = Coord2D::new(x, y);
                match state_diff.get_tile(position).unwrap() {
                    Tile::Grass(0) => (),
                    Tile::Grass(3) => {
                        let expand =
                            handle_tick % WORLD_GRASS_EXPAND_CYCLE_DURATION < WORLD_TASK_DURATION;
                        if expand {
                            let side = (handle_tick / WORLD_GRASS_EXPAND_CYCLE_DURATION) % 4;
                            if let Some(neighbor) = match side {
                                0 => Some((x - 1, y)),
                                1 => Some((x + 1, y)),
                                2 => Some((x, y - 1)),
                                3 => Some((x, y + 1)),
                                _ => None,
                            } {
                                let neighbor_position = Coord2D::from_tuple(neighbor);
                                if let Some(Tile::Grass(0)) = state_diff.get_tile(neighbor_position)
                                {
                                    state_diff.set_tile(neighbor_position, Tile::Grass(1))
                                }
                            }
                        }
                    }
                    Tile::Grass(amount) => {
                        if handle_tick % WORLD_GRASS_GROW_CYCLE_DURATION < WORLD_TASK_DURATION {
                            state_diff.set_tile(position, Tile::Grass(amount + 1))
                        }
                    }
                    Tile::Obstacle => (),
                }
            }
        }

        Some(Box::new(WorldStep))
    }

    fn display_action(&self) -> DisplayAction {
        DisplayAction::WorldStep
    }

    fn is_valid(
        &self,
        _tick: u64,
        _state_diff: StateDiffRef<EcosystemDomain>,
        _agent: AgentId,
    ) -> bool {
        true
    }

    impl_task_boxed_methods!(EcosystemDomain);
}
