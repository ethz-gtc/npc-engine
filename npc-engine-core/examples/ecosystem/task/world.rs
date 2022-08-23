/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use npc_engine_core::{impl_task_boxed_methods, Context, ContextMut, Task, TaskDuration};
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
    fn duration(&self, _ctx: Context<EcosystemDomain>) -> TaskDuration {
        WORLD_TASK_DURATION
    }

    fn execute(
        &self,
        mut ctx: ContextMut<EcosystemDomain>,
    ) -> Option<Box<dyn Task<EcosystemDomain>>> {
        // 1. Consume food, every time, and make kids, if it is the right time
        for agent in ctx.state_diff.list_agents() {
            // Consume food
            let state = ctx.state_diff.get_agent_mut(agent).unwrap();
            if state.food > 0 {
                state.food -= 1;
                if state.food == 0 {
                    state.kill(ctx.tick)
                }
            }
            // Make baby
            // Note: It is important that only a single agent creates new agent, otherwise
            // there is a risk of collision in AgentId. In that case, either an atomic could be used,
            // but the agent id space will be consumed fast, or some form of agent id renaming will
            // need to be implemented.
            let baby_season =
                ctx.tick % state.ty.reproduction_cycle_duration() < WORLD_TASK_DURATION;
            if baby_season && state.food >= state.ty.min_food_for_reproduction() {
                let parent_position = state.position;
                let parent_ty = state.ty;
                // if agent of same type around us, too crowded, do not reproduce
                let mut alone = true;
                for direction in DIRECTIONS {
                    let neighbor_position = DirConv::apply(direction, parent_position);
                    if ctx
                        .state_diff
                        .get_agent_at(neighbor_position)
                        .map_or(false, |(_, agent)| agent.ty == parent_ty)
                    {
                        alone = false;
                        break;
                    }
                }
                // if alone, attempt to reproduce
                if alone {
                    for direction in DIRECTIONS {
                        let baby_position = DirConv::apply(direction, parent_position);
                        if ctx.state_diff.is_position_free(baby_position) {
                            // modify parent
                            let parent_state = ctx.state_diff.get_agent_mut(agent).unwrap();
                            let food = parent_state.ty.food_given_to_baby();
                            parent_state.food -= food;
                            // create child
                            ctx.state_diff.new_agent(AgentState {
                                ty: parent_ty,
                                birth_date: ctx.tick,
                                position: baby_position,
                                food,
                                death_date: None,
                            });
                            break;
                        }
                    }
                }
            }
        }
        // 2. Grow grass, if it is time (1/WORLD_GRASS_REGROW_PERIOD of times)
        let width = ctx.state_diff.map_width();
        let height = ctx.state_diff.map_height();
        for y in 0..height {
            for x in 0..width {
                let handle_tick = ctx
                    .tick
                    .wrapping_add((x as u64) * 2797)
                    .wrapping_add((y as u64) * 3637);
                let position = Coord2D::new(x, y);
                match ctx.state_diff.get_tile(position).unwrap() {
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
                                if let Some(Tile::Grass(0)) =
                                    ctx.state_diff.get_tile(neighbor_position)
                                {
                                    ctx.state_diff.set_tile(neighbor_position, Tile::Grass(1))
                                }
                            }
                        }
                    }
                    Tile::Grass(amount) => {
                        if handle_tick % WORLD_GRASS_GROW_CYCLE_DURATION < WORLD_TASK_DURATION {
                            ctx.state_diff.set_tile(position, Tile::Grass(amount + 1))
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

    fn is_valid(&self, _ctx: Context<EcosystemDomain>) -> bool {
        true
    }

    impl_task_boxed_methods!(EcosystemDomain);
}
