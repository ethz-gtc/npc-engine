/* 
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich, see AUTHORS.txt for details
 */

use std::collections::{BTreeSet, HashMap, BTreeMap};

use npc_engine_common::{Domain, Behavior, AgentValue, AgentId, StateDiffRef};
use npc_engine_utils::{Direction, GlobalDomain, DomainWithPlanningTask, Coord2D};
use num_traits::Zero;

use crate::{state::{GlobalState, LocalState, Diff, Access}, behavior::{herbivore::Herbivore, world::{WorldBehavior, WORLD_AGENT_ID}}, map::{GridAccess, Map}};

type PreyDistance = u8;

#[derive(Debug)]
pub enum DisplayAction {
	Idle,
	Plan,
	Move(Direction),
	Jump(Direction),
	EatGrass,
	EatPrey(PreyDistance),
	WorldStep
}
impl Default for DisplayAction {
    fn default() -> Self {
        DisplayAction::Idle
    }
}

pub struct EcosystemDomain;

impl Domain for EcosystemDomain {
	type State = LocalState;
	type Diff = Diff;
	type DisplayAction = DisplayAction;

	fn list_behaviors() -> &'static [&'static dyn Behavior<Self>] {
		&[&Herbivore, &WorldBehavior]
	}

	fn get_current_value(tick: u64, state_diff: StateDiffRef<Self>, agent: AgentId) -> AgentValue {
		state_diff.get_agent(agent)
			.map_or(AgentValue::zero(), |agent_state| {
				if agent_state.alive {
					let age = tick - agent_state.birth_date;
					AgentValue::new(age as f32).unwrap()
				} else {
					AgentValue::zero()
				}
			})
	}

	fn update_visible_agents(_start_tick: u64, _tick: u64, state_diff: StateDiffRef<Self>, _agent: AgentId, agents: &mut BTreeSet<AgentId>) {
		// clear the list
		agents.clear();
		// add all agents from the state
		agents.extend(state_diff.initial_state.agents.keys());
		// remove dead agents
		for (agent, agent_state) in state_diff.diff.agents.iter() {
			if !agent_state.alive {
				agents.remove(agent);
			}
		}
		// add world agent
		agents.insert(WORLD_AGENT_ID);
    }

	fn get_state_description(state_diff: StateDiffRef<Self>) -> String {
		let mut agents = BTreeMap::new();
		for (agent, agent_state) in state_diff.initial_state.agents.iter() {
			agents.insert(*agent, agent_state.clone());
		}
		for (agent, agent_state) in state_diff.diff.agents.iter() {
			if !agents.contains_key(agent) {
				agents.insert(*agent, agent_state.clone());
			}
		}
		let mut s = String::new();
		for (agent, agent_state) in agents.into_iter() {
			let food = if agent_state.alive {
				format!("🍞{}", agent_state.food)
			} else {
				"🕇".into()
			};
			s += &format!("{}:{}{}, ", agent.0, agent_state.position, food);
		}
		s
	}

	fn display_action_task_planning() -> Self::DisplayAction {
		DisplayAction::Plan
	}
}

const AGENTS_RADIUS: i32 = 4;
const MAP_RADIUS: i32 = 8;

fn derive_local_state_radius(global_state: &GlobalState, agent: AgentId, map_radius: i32, agent_radius: i32) -> LocalState {
	if agent == WORLD_AGENT_ID {
		// collect all agents
		LocalState {
			origin: Coord2D::default(),
			map: Map::empty(),
			agents: global_state.agents.clone()
		}
	} else {
		let agent_state = global_state.agents.get(&agent).unwrap();
		// extract tiles
		let (origin, map) = global_state.map.extract_region(agent_state.position, map_radius);
		// extract agents
		let agents = global_state.get_agents_in_region(agent_state.position, agent_radius)
			.map(|(agent, agent_state)| {
				let mut agent_state = agent_state.clone();
				agent_state.position -= origin;
				(*agent, agent_state)
			})
			.collect();
		LocalState {
			origin,
			map,
			agents,
		}
	}
}

impl GlobalDomain for EcosystemDomain {
	type GlobalState = GlobalState;

	fn derive_local_state(global_state: &Self::GlobalState, agent: AgentId) -> Self::State {
		derive_local_state_radius(global_state, agent, MAP_RADIUS, AGENTS_RADIUS)
    }

	fn apply(global_state: &mut Self::GlobalState, local_state: &Self::State, diff: &Self::Diff) {
		// update tiles
		for (&pos, &tile) in diff.map.tiles.iter() {
			*global_state.map.at_mut(pos + local_state.origin).unwrap() = tile;
		}
		// update agents
		for (agent, mut agent_state) in diff.agents.iter().cloned() {
			agent_state.position += local_state.origin;
			global_state.agents.insert(agent, agent_state);
		}
    }
}

impl DomainWithPlanningTask for EcosystemDomain {}

#[cfg(test)]
mod tests {
	use std::{str::FromStr, collections::HashMap};

	use crate::*;
	use super::*;

	fn create_test_global_state() -> GlobalState {
		let map = Map::from_str(
			"#0000\n\
			 01230\n\
			 ###00"
		).unwrap();
		let agents = HashMap::from([
			(
				AgentId(1),
				AgentState {
					ty: AgentType::Herbivore,
					birth_date: 0,
					position: Coord2D::new(1, 0),
					food: 2,
					alive: true
				}
			),
			(
				AgentId(3),
				AgentState {
					ty: AgentType::Carnivore,
					birth_date: 2,
					position: Coord2D::new(3, 2),
					food: 5,
					alive: true
				}
			),
		]);
		GlobalState {
			map,
			agents,
		}
	}

	#[test]
	fn global_domain() {
		let global_state = create_test_global_state();
		let local_state = derive_local_state_radius(&global_state, AgentId(1), 2, 1);
		assert_eq!(local_state.agents.len(), 1);
		assert!(local_state.agents.get(&AgentId(1)).is_some());
		assert_eq!(local_state.map.size(), Coord2D::new(4, 3));
	}
}