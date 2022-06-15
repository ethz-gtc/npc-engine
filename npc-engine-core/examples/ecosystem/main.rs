/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use core::time;
#[allow(unused_imports)]
use std::collections::HashMap;
use std::{collections::HashSet, iter, num::NonZeroU64, time::Duration};

use behavior::world::WORLD_AGENT_ID;
use constants::*;
use domain::EcosystemDomain;
use map::{GridAccess, Map, Tile};
use npc_engine_core::{ActiveTask, ActiveTasks, AgentId, IdleTask, MCTSConfiguration, MCTS};
use npc_engine_utils::{
    plot_tree_in_tmp_with_task_name, run_threaded_executor, Coord2D, ExecutorState,
    ExecutorStateGlobal,
};
use rand::Rng;
use state::{Agents, Diff, GlobalState, LocalState};
use task::{eat_grass::EatGrass, eat_herbivore::EatHerbivore, world::WorldStep};

use crate::state::{AgentState, AgentType};

mod behavior;
mod constants;
mod domain;
mod map;
mod state;
mod task;

#[derive(Default, Debug, Clone)]
struct EcosystemExecutorState {
    herbivore_eat_count: u32,
    carnivore_eat_count: u32,
}
impl ExecutorStateGlobal<EcosystemDomain> for EcosystemExecutorState {
    fn create_initial_state(&self) -> GlobalState {
        let mut map = Map::new(MAP_SIZE, Tile::Grass(0));

        // helper for terrain
        let mut add_random_and_hotspots =
            |random_count, hotspot_count, tile_factory: &dyn Fn() -> Tile| {
                for _ in 0..random_count {
                    let pos = Coord2D::rand_uniform(MAP_SIZE);
                    *map.at_mut(pos).unwrap() = tile_factory();
                }
                let mut rng = rand::thread_rng();
                for _ in 0..hotspot_count {
                    let x = rng.gen_range(2..MAP_SIZE.x - 2);
                    let y = rng.gen_range(2..MAP_SIZE.y - 2);
                    let pos = Coord2D::new(x, y);
                    for _ in 0..10 {
                        let x = rng.gen_range(pos.x - 2..=pos.x + 2);
                        let y = rng.gen_range(pos.y - 2..=pos.y + 2);
                        let pos = Coord2D::new(x, y);
                        *map.at_mut(pos).unwrap() = tile_factory();
                    }
                }
            };

        // obstacles
        add_random_and_hotspots(OBSTACLE_RANDOM_COUNT, OBSTACLE_HOTSPOT_COUNT, &|| {
            Tile::Obstacle
        });

        // plants
        add_random_and_hotspots(PLANT_RANDOM_COUNT, PLANT_HOTSPOT_COUNT, &|| {
            let mut rng = rand::thread_rng();
            Tile::Grass(rng.gen_range(0..=3))
        });

        // helper for animals
        let mut agents = Agents::new();
        let mut used_poses = HashSet::new();
        let mut agent_id = 0;
        let mut add_animals = |ty, count, food| {
            for _i in 0..count {
                loop {
                    let pos = Coord2D::rand_uniform(MAP_SIZE);
                    if !used_poses.contains(&pos) && *map.at(pos).unwrap() != Tile::Obstacle {
                        used_poses.insert(pos);
                        agents.insert(
                            AgentId(agent_id),
                            AgentState {
                                ty,
                                birth_date: 0,
                                position: pos,
                                food,
                                alive: true,
                            },
                        );
                        agent_id += 1;
                        break;
                    }
                }
            }
        };

        // animals
        add_animals(AgentType::Herbivore, HERBIVORE_COUNT, HERBIVORE_MAX_FOOD);
        add_animals(AgentType::Carnivore, CARNIVORE_COUNT, CARNIVORE_MAX_FOOD);
        //*map.at_mut(Coord2D::new(0, 1)).unwrap() = Tile::Obstacle;
        //*map.at_mut(Coord2D::new(0, 0)).unwrap() = Tile::Grass(3);
        // let agents = HashMap::from([
        // 	(
        // 		AgentId(0),
        // 		AgentState {
        // 			ty: AgentType::Herbivore,
        // 			birth_date: 0,
        // 			position: Coord2D::new(0, 0),
        // 			food: 3,
        // 			alive: true
        // 		}
        // 	),
        // 	(
        // 		AgentId(1),
        // 		AgentState {
        // 			ty: AgentType::Carnivore,
        // 			birth_date: 0,
        // 			position: Coord2D::new(2, 0),
        // 			food: 5,
        // 			alive: true
        // 		}
        // 	)
        // ]);
        GlobalState { map, agents }
    }

    fn init_task_queue(&self, state: &GlobalState) -> ActiveTasks<EcosystemDomain> {
        state
            .agents
            .iter()
            .map(|(id, _)| ActiveTask::new_with_end(0, *id, Box::new(IdleTask)))
            .chain(iter::once(ActiveTask::new_with_end(
                10,
                WORLD_AGENT_ID,
                Box::new(WorldStep),
            )))
            .collect()
    }

    fn keep_agent(&self, _tick: u64, state: &GlobalState, agent: AgentId) -> bool {
        agent == WORLD_AGENT_ID
            || state
                .agents
                .get(&agent)
                .map_or(false, |agent_state| agent_state.alive)
    }

    fn keep_execution(
        &self,
        _tick: u64,
        queue: &ActiveTasks<EcosystemDomain>,
        _state: &GlobalState,
    ) -> bool {
        queue.len() > 1
    }

    fn post_step_hook(&self, _tick: u64, state: &GlobalState) {
        print!(
            "\x1B[H\
            🐄: {}🌿, 🐅: {}🍖\n\
            {}",
            self.herbivore_eat_count, self.carnivore_eat_count, *state
        );
    }
}

impl ExecutorState<EcosystemDomain> for EcosystemExecutorState {
    fn post_action_execute_hook(
        &mut self,
        _state: &LocalState,
        _diff: &Diff,
        active_task: &ActiveTask<EcosystemDomain>,
        _queue: &mut ActiveTasks<EcosystemDomain>,
    ) {
        let task = &active_task.task;
        if task.downcast_ref::<EatGrass>().is_some() {
            self.herbivore_eat_count += 1;
        }
        if task.downcast_ref::<EatHerbivore>().is_some() {
            self.carnivore_eat_count += 1;
        }
    }
    fn post_mcts_run_hook(
        &mut self,
        mcts: &MCTS<EcosystemDomain>,
        last_active_task: &ActiveTask<EcosystemDomain>,
    ) {
        if let Err(e) = plot_tree_in_tmp_with_task_name(mcts, "ecosystem", last_active_task) {
            println!("Cannot write search tree: {e}");
        }
    }
}

fn main() {
    // These parameters control the MCTS algorithm.
    let mcts_config = MCTSConfiguration {
        allow_invalid_tasks: false,
        visits: 1000,
        depth: 50,
        exploration: 1.414,
        discount_hl: 17.,
        seed: None,
        planning_task_duration: Some(NonZeroU64::new(3).unwrap()),
    };

    // Enable logging if specified in the RUST_LOG environment variable.
    env_logger::init();

    // First clear the screen.
    clearscreen::clear().unwrap();

    // State of the execution.
    let mut executor_state = EcosystemExecutorState::default();

    // Run as long as there is at least one agent alive.
    const ONE_FRAME: Duration = time::Duration::from_millis(40);
    run_threaded_executor(&mcts_config, &mut executor_state, ONE_FRAME);
}
