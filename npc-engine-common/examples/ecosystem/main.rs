use core::time;
use std::{collections::HashSet, num::NonZeroU64, thread, iter};

use behavior::world::WORLD_AGENT_ID;
use domain::EcosystemDomain;
use map::{Tile, Map, GridAccess};
use npc_engine_common::{AgentId, ActiveTasks, ActiveTask, IdleTask, MCTSConfiguration, MCTS};
use npc_engine_utils::{Coord2D, ExecutorStateGlobal, ExecutorState, ThreadedExecutor, plot_tree_in_tmp_with_task_name};
use rand::Rng;
use state::{GlobalState, Agents};
use task::world::WorldStep;

use crate::state::{AgentState, AgentType};

mod map;
mod state;
mod domain;
mod task;
mod behavior;

const MAP_SIZE: Coord2D = Coord2D::new(40, 20);
// const MAP_SIZE: Coord2D = Coord2D::new(4, 4);
const OBSTACLE_RANDOM_COUNT: usize = 20;
const OBSTACLE_HOTSPOT_COUNT: usize = 6;
const PLANT_RANDOM_COUNT: usize = 40;
const PLANT_HOTSPOT_COUNT: usize = 9;
const HERBIVORE_COUNT: usize = 10;

struct EcosystemExecutorState;
impl ExecutorStateGlobal<EcosystemDomain> for EcosystemExecutorState {
    fn create_initial_state(&self) -> GlobalState {
		let mut map = Map::new(MAP_SIZE, Tile::Grass(0));
		let mut random_and_hotspots =
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
		random_and_hotspots(OBSTACLE_RANDOM_COUNT, OBSTACLE_HOTSPOT_COUNT, &|| Tile::Obstacle);
		// plants
		random_and_hotspots(PLANT_RANDOM_COUNT, PLANT_HOTSPOT_COUNT, &|| {
			let mut rng = rand::thread_rng();
			Tile::Grass(rng.gen_range(0..=3))
		});
		// herbivores
		let mut used_poses = HashSet::new();
		let mut agents = Agents::new();
		let mut agent_id = 0;
		for _i in 0..HERBIVORE_COUNT {
			loop {
				let pos = Coord2D::rand_uniform(MAP_SIZE);
				if !used_poses.contains(&pos) && *map.at(pos).unwrap() != Tile::Obstacle {	
					used_poses.insert(pos);
					agents.insert(
						AgentId(agent_id),
						AgentState {
							ty: AgentType::Herbivore,
							birth_date: 0,
							position: pos,
							food: 3,
							alive: true
						}
					);
					agent_id += 1;
					break;
				}
			}
		}
		GlobalState {
			map,
			agents
		}
    }

	fn init_task_queue(&self, state: &GlobalState) -> ActiveTasks<EcosystemDomain> {
        state.agents.iter()
			.map(|(id, _)|
				ActiveTask::new_with_end(0, *id, Box::new(IdleTask))
			)
			.chain(iter::once(
				ActiveTask::new_with_end(10, WORLD_AGENT_ID, Box::new(WorldStep))
			))
			.collect()
    }

	fn keep_agent(&self, _tick: u64, state: &GlobalState, agent: AgentId) -> bool {
		agent == WORLD_AGENT_ID ||
		state.agents.get(&agent)
			.map_or(false, |agent_state| agent_state.alive)
	}
}

impl ExecutorState<EcosystemDomain> for EcosystemExecutorState {
	fn post_mcts_run_hook(&mut self, mcts: &MCTS<EcosystemDomain>, last_active_task: &ActiveTask<EcosystemDomain>) {
		if let Err(e) = plot_tree_in_tmp_with_task_name(mcts, "ecosystem", last_active_task) {
			println!("Cannot write search tree: {e}");
		}
	}
}

fn main() {
	env_logger::init();
	let mcts_config = MCTSConfiguration {
		allow_invalid_tasks: false,
		visits: 1000,
		depth: 50,
		exploration: 1.414,
		discount_hl: 17.,
		seed: None,
		planning_task_duration: Some(NonZeroU64::new(3).unwrap()),
	};
	let mut executor_state = EcosystemExecutorState;
	let mut executor = ThreadedExecutor::new(
		mcts_config,
		&mut executor_state
	);
	let one_frame = time::Duration::from_millis(40);
	while executor.agents_count() > 1 {
		executor.step();
		thread::sleep(one_frame);
		clearscreen::clear().unwrap();
		print!("{}", executor.state());
	}
}