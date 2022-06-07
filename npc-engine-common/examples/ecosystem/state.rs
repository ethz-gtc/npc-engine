/* 
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich, see AUTHORS.txt for details
 */

use std::{collections::{HashMap, BTreeMap}, fmt::Display};

use npc_engine_common::{AgentId, StateDiffRef, StateDiffRefMut};
use npc_engine_utils::{Coord2D, keep_second_mut};

use crate::{map::{Map, Tile, GridAccess}, domain::EcosystemDomain};

#[derive(Clone, Copy, Eq, PartialEq, Debug, Hash)]
pub enum AgentType {
    Herbivore,
    Carnivore,
}

#[derive(Debug, Hash, Clone, Eq, PartialEq)]
pub struct AgentState {
	pub ty: AgentType,
	pub birth_date: u64,
	pub position: Coord2D,
	pub food: u32,
	pub alive: bool
}
impl AgentState {

}
pub type Agents = HashMap<AgentId, AgentState>;

fn format_map_and_agents(f: &mut std::fmt::Formatter<'_>, map: &Map, agents: &Agents) -> std::fmt::Result {
	// collect agents into a map indexed by locations
	let mut agents_map = HashMap::new();
	for (_, agent_state) in agents.iter() {
		agents_map.entry(agent_state.position)
			.or_insert_with(Vec::new)
			.push((agent_state.ty, agent_state.alive));
	}
	// iterate positions, showing agents if needed
	for (y, row) in map.0.iter().enumerate() {
		for (x, tile) in row.iter().enumerate() {
			use ansi_term::Colour::Fixed;
			let background = Fixed(match *tile {
				Tile::Obstacle => 242,
				Tile::Grass(0) => 58,
				Tile::Grass(1) => 148,
				Tile::Grass(2) => 154,
				Tile::Grass(_) => 40,
			});
			let pos = Coord2D::new(x as i32, y as i32);
			let text = if let Some(agents) = agents_map.get(&pos) {
				let mut best: Option<(_, bool)> = None;
				for (ty, alive) in agents.iter().copied() {
					if let Some((_, that_alive)) = best {
						if alive || !that_alive {
							best = Some((ty, alive));
						}
					} else {
						best = Some((ty, alive));
					}
				}
				match best {
					Some((AgentType::Herbivore, true)) => "🐄",
					Some((AgentType::Carnivore, true)) => "🐅",
					Some((AgentType::Herbivore, false)) => "☠️",
					Some((AgentType::Carnivore, false)) => "💀",
					None => "  ",
				}
			} else {
				"  "
			};
			write!(f, "{}", Fixed(0).on(background).paint(text))?;
		}
		writeln!(f)?;
	}
	Ok(())
}

#[derive(Debug, Clone)]
pub struct GlobalState {
	pub map: Map,
	pub agents: Agents
}
impl GlobalState {
	pub fn get_agents_in_region(&self, center: Coord2D, radius: i32) -> impl Iterator<Item=(&AgentId, &AgentState)> {
		self.agents.iter()
			.filter(move |(_, agent_state)| {
				agent_state.position.largest_dim_dist(&center) <= radius
			})
	}
}
impl Display for GlobalState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        format_map_and_agents(f, &self.map, &self.agents)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct LocalState {
	pub origin: Coord2D,
	pub map: Map,
	pub agents: Agents
}
impl Display for LocalState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        format_map_and_agents(f, &self.map, &self.agents)
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct MapDiff {
    pub tiles: BTreeMap<Coord2D, Tile>
}
impl MapDiff {
	fn new(tiles: BTreeMap<Coord2D, Tile>) -> Self {
		Self {
			tiles
		}
	}
}
impl Default for MapDiff {
    fn default() -> Self {
        Self::new(BTreeMap::new())
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Default)]
pub struct Diff {
	pub map: MapDiff,
	pub agents: Vec<(AgentId, AgentState)>
}
impl Diff {
	fn has_agent(&self, agent: AgentId) -> bool {
		self.get_agent(agent).is_some()
	}
	fn get_agent(&self, agent: AgentId) -> Option<&AgentState> {
		self.agents.iter()
			.find_map(|(id, state)|
				if *id == agent {
					Some(state)
				} else {
					None
				}
			)
	}
	fn get_agent_mut(&mut self, agent: AgentId) -> Option<&mut AgentState> {
		self.agents.iter_mut()
			.find_map(|(id, state)|
				if *id == agent {
					Some(state)
				} else {
					None
				}
			)
	}
}

pub trait Access {
    fn get_tile(&self, position: Coord2D) -> Option<Tile>;
	fn get_grass(&self, position: Coord2D) -> Option<u8> {
		self.get_tile(position)
			.and_then(|tile|
				if let Tile::Grass(growth) = tile {
					Some(growth)
				} else {
					None
				}
			)
	}
    fn get_agent(&self, agent: AgentId) -> Option<&AgentState>;
    fn get_agent_at(&self, position: Coord2D) -> Option<(AgentId, &AgentState)>;
    fn get_first_adjacent_agent(&self, position: Coord2D, n: u8) -> Option<(AgentId, &AgentState)> {
		self.get_agent_at(position + Coord2D::new(-(n as i32), 0))
			.or_else(|| self.get_agent_at(position + Coord2D::new(n as i32, 0)))
			.or_else(|| self.get_agent_at(position + Coord2D::new(0, n as i32)))
			.or_else(|| self.get_agent_at(position + Coord2D::new(0, -(n as i32))))
    }
}

impl Access for StateDiffRef<'_, EcosystemDomain> {
    fn get_tile(&self, position: Coord2D) -> Option<Tile> {
		self.diff.map.tiles
			.get(&position)
			.or_else(||
				self.initial_state.map.at(position)
			)
			.copied()
    }

    fn get_agent(&self, agent: AgentId) -> Option<&AgentState> {
        self.diff
			.get_agent(agent)
			.or_else(||
				self.initial_state.agents.get(&agent)
			)
    }

    fn get_agent_at(&self, position: Coord2D) -> Option<(AgentId, &AgentState)> {
		fn filter_position<'l>(position: Coord2D, id: AgentId, state: &'l AgentState) -> Option<(AgentId, &'l AgentState)> {
			if state.position == position {
				Some((id, state))
			} else {
				None
			}
		}
		self.diff.agents.iter()
			.find_map(|(id, state)|
				filter_position(position, *id, state)
			)
			.or_else(||
				self.initial_state.agents.iter()
					.find_map(|(id, state)|
						if self.diff.get_agent(*id).is_some() {
							None
						} else {
							filter_position(position, *id, state)
						}
					)
			)
    }
}

pub trait AccessMut {
    fn get_agent_mut(&mut self, agent: AgentId) -> Option<&mut AgentState>;
    fn set_tile(&mut self, position: Coord2D, tile: Tile);
    fn get_agent_pos_mut(&mut self, agent: AgentId) -> Option<&mut Coord2D> {
		self.get_agent_mut(agent)
			.map(|state| &mut state.position)
	}
}

impl AccessMut for StateDiffRefMut<'_, EcosystemDomain> {
    fn get_agent_mut(&mut self, agent: AgentId) -> Option<&mut AgentState> {
		if self.diff.has_agent(agent) {
			self.diff.get_agent_mut(agent)
		} else {
			self.initial_state.agents.get(&agent).and_then(|agent_state| {
				self.diff.agents.push((agent, agent_state.clone()));
				self.diff.agents.last_mut()
					.map(keep_second_mut)
			})
		}
    }

    fn set_tile(&mut self, position: Coord2D, tile: Tile) {
		if *self.initial_state.map.at(position).unwrap() == tile {
			self.diff.map.tiles.remove(&position);
		} else {
			self.diff.map.tiles.insert(position, tile);
		}
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use crate::*;
    use super::*;

	fn create_test_local_state() -> LocalState {
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
		LocalState {
			origin: Coord2D::default(),
			map,
			agents,
		}
	}

    #[test]
	fn access() {
		let state = create_test_local_state();
		// empty diff
		let diff = Diff::default();
		let state_diff = StateDiffRef::new(&state, &diff);
		assert_eq!(state_diff.get_tile(Coord2D::new(0, 0)).unwrap(), Tile::Obstacle);
		assert_eq!(state_diff.get_tile(Coord2D::new(4, 2)).unwrap(), Tile::Grass(0));
		assert_eq!(state_diff.get_tile(Coord2D::new(-1, -1)), None);
		assert_eq!(state_diff.get_tile(Coord2D::new(5, 0)), None);
		assert_eq!(state_diff.get_tile(Coord2D::new(0, 4)), None);
		assert_eq!(state_diff.get_grass(Coord2D::new(0, 0)), None);
		assert_eq!(state_diff.get_grass(Coord2D::new(3, 1)).unwrap(), 3);
		assert_eq!(state_diff.get_agent(AgentId(0)), None);
		assert_eq!(state_diff.get_agent(AgentId(1)).unwrap().position, Coord2D::new(1, 0));
		assert_eq!(state_diff.get_agent(AgentId(3)).unwrap().position, Coord2D::new(3, 2));
		assert_eq!(state_diff.get_agent_at(Coord2D::new(2, 2)), None);
		assert_eq!(state_diff.get_agent_at(Coord2D::new(1, 0)).unwrap().0, AgentId(1));
		assert_eq!(state_diff.get_first_adjacent_agent(Coord2D::new(1, 2), 1), None);
		assert_eq!(state_diff.get_first_adjacent_agent(Coord2D::new(1, 2), 2).unwrap().0, AgentId(3));
		// diff with some changes
		let diff = Diff {
			map: MapDiff::new(BTreeMap::from([
				(Coord2D::new(1, 0), Tile::Grass(1)),
				(Coord2D::new(3, 1), Tile::Grass(2)),
			])),
			agents: vec![
				(AgentId(3), AgentState {
					ty: AgentType::Carnivore,
					birth_date: 2,
					position: Coord2D::new(3, 1),
					food: 6,
					alive: true
				})
			],
		};
		let state_diff = StateDiffRef::new(&state, &diff);
		assert_eq!(state_diff.get_tile(Coord2D::new(0, 0)).unwrap(), Tile::Obstacle);
		assert_eq!(state_diff.get_tile(Coord2D::new(1, 0)).unwrap(), Tile::Grass(1));
		assert_eq!(state_diff.get_tile(Coord2D::new(3, 1)).unwrap(), Tile::Grass(2));
		assert_eq!(state_diff.get_grass(Coord2D::new(3, 1)).unwrap(), 2);
		assert_eq!(state_diff.get_agent(AgentId(1)).unwrap().position, Coord2D::new(1, 0));
		assert_eq!(state_diff.get_agent(AgentId(3)).unwrap().position, Coord2D::new(3, 1));
		assert_eq!(state_diff.get_agent_at(Coord2D::new(3, 1)).unwrap().0, AgentId(3));
		assert_eq!(state_diff.get_agent_at(Coord2D::new(3, 2)), None);
		assert_eq!(state_diff.get_first_adjacent_agent(Coord2D::new(1, 2), 2).unwrap().0, AgentId(1));
		// mutable access
		let mut diff = Diff::default();
		let mut state_diff_mut = StateDiffRefMut::new(&state, &mut diff);
		state_diff_mut.get_agent_mut(AgentId(1)).unwrap().food = 3;
		assert_eq!(diff.agents.len(), 1);
		assert_eq!(diff.agents[0].0, AgentId(1));
		assert_eq!(diff.agents[0].1.food, 3);
		let mut state_diff_mut = StateDiffRefMut::new(&state, &mut diff);
		*state_diff_mut.get_agent_pos_mut(AgentId(1)).unwrap() = Coord2D::new(2, 0);
		assert_eq!(diff.agents[0].1.position, Coord2D::new(2, 0));
		let mut state_diff_mut = StateDiffRefMut::new(&state, &mut diff);
		state_diff_mut.set_tile(Coord2D::new(1, 0), Tile::Grass(0));
		state_diff_mut.set_tile(Coord2D::new(4, 2), Tile::Grass(2));
		assert!(!diff.map.tiles.contains_key(&Coord2D::new(1, 0)));
		assert_eq!(*diff.map.tiles.get(&Coord2D::new(4, 2)).unwrap(), Tile::Grass(2));
	}

	#[test]
	fn get_agents_in_region() {
		let state = create_test_local_state();
		let state = GlobalState {
			map: state.map,
			agents: state.agents,
		};
		let get_agents = |center, radius| state
			.get_agents_in_region(center, radius)
			.collect::<Vec<_>>()
		;
		let agents = get_agents(Coord2D::new(0, 0), 0);
		assert_eq!(agents.len(), 0);
		let agents = get_agents(Coord2D::new(0, 0), 1);
		assert_eq!(agents.len(), 1);
		let agents = get_agents(Coord2D::new(1, 0), 0);
		assert_eq!(agents.len(), 1);
		let agents = get_agents(Coord2D::new(0, 0), 2);
		assert_eq!(agents.len(), 1);
		let agents = get_agents(Coord2D::new(2, 1), 1);
		assert_eq!(agents.len(), 2);
	}
}