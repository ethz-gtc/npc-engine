/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::collections::{BTreeMap, BTreeSet, HashMap};

use npc_engine_core::{
    AgentId, AgentValue, Behavior, Context, Domain, DomainWithPlanningTask, StateDiffRef,
};
use npc_engine_utils::{Coord2D, Direction, GlobalDomain};
use num_traits::Zero;

use crate::{
    behavior::{
        carnivore::Carnivore,
        herbivore::Herbivore,
        world::{WorldBehavior, WORLD_AGENT_ID},
    },
    constants::*,
    map::GridAccess,
    state::{Access, AgentType, Diff, GlobalState, LocalState},
};

#[derive(Debug, Default)]
#[allow(dead_code)]
pub enum DisplayAction {
    #[default]
    Idle,
    Plan,
    Move(Direction),
    Jump(Direction),
    EatGrass,
    EatHerbivore(Direction),
    WorldStep,
}

pub struct EcosystemDomain;

impl Domain for EcosystemDomain {
    type State = LocalState;
    type Diff = Diff;
    type DisplayAction = DisplayAction;

    fn list_behaviors() -> &'static [&'static dyn Behavior<Self>] {
        &[&Herbivore, &Carnivore, &WorldBehavior]
    }

    fn get_current_value(tick: u64, state_diff: StateDiffRef<Self>, agent: AgentId) -> AgentValue {
        state_diff
            .get_agent(agent)
            .map_or(AgentValue::zero(), |agent_state| {
                if agent_state.alive() {
                    let age = tick - agent_state.birth_date;
                    AgentValue::new(age as f32).unwrap()
                } else {
                    AgentValue::zero()
                }
            })
    }

    fn update_visible_agents(
        _start_tick: u64,
        ctx: Context<EcosystemDomain>,
        agents: &mut BTreeSet<AgentId>,
    ) {
        // clear the list
        agents.clear();
        // add all agents from the state
        agents.extend(ctx.state_diff.initial_state.agents.keys());
        // remove dead agents
        for (agent, agent_state) in ctx.state_diff.diff.cur_agents.iter() {
            if !agent_state.alive() {
                agents.remove(agent);
            }
        }
        // add alive new agents only in the diff
        for (agent, agent_state) in ctx.state_diff.diff.new_agents.iter() {
            if agent_state.alive() {
                agents.insert(*agent);
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
        for (agent, agent_state) in state_diff.diff.cur_agents.iter() {
            if !agents.contains_key(agent) {
                agents.insert(*agent, agent_state.clone());
            }
        }
        let mut s = String::new();
        for (agent, agent_state) in agents.into_iter() {
            let food = if agent_state.alive() {
                format!("🍞{}", agent_state.food)
            } else {
                "🕇".into()
            };
            use std::fmt::Write;
            write!(s, "{}:{}{}, ", agent.0, agent_state.position, food).unwrap();
        }
        s
    }

    fn get_new_agents(state_diff: StateDiffRef<Self>) -> Vec<AgentId> {
        state_diff.diff.get_new_agents_ids().collect()
    }

    fn display_action_task_planning() -> Self::DisplayAction {
        DisplayAction::Plan
    }
}

fn derive_local_state_radius(
    global_state: &GlobalState,
    agent: AgentId,
    map_radius: i32,
    agent_radius: i32,
) -> LocalState {
    assert!(agent != WORLD_AGENT_ID);
    let agent_state = global_state.agents.get(&agent).unwrap();
    let center_position = agent_state.position;
    // extract tiles
    let (origin, map) = global_state.map.extract_region(center_position, map_radius);
    // extract alive agents, limiting the number to MAX_AGENTS_ATTENTION closest agents
    let mut agents = global_state
        .get_agents_in_region(center_position, agent_radius)
        .filter_map(|(agent, agent_state)| {
            let mut agent_state = agent_state.clone();
            if agent_state.alive() {
                let dist = agent_state.position.manhattan_dist(center_position);
                agent_state.position -= origin;
                Some((dist, (*agent, agent_state)))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    agents.sort_by(|a, b| a.0.cmp(&b.0));
    let agents = agents
        .into_iter()
        .take(MAX_AGENTS_ATTENTION)
        .map(|(_, agent_and_state)| agent_and_state)
        .collect::<HashMap<_, _>>();
    LocalState {
        origin,
        map,
        agents,
        next_agent_id: global_state.next_agent_id,
    }
}

impl GlobalDomain for EcosystemDomain {
    type GlobalState = GlobalState;

    fn derive_local_state(global_state: &Self::GlobalState, agent: AgentId) -> Self::State {
        global_state.agents.get(&agent).map_or(
            // world agent, collect all agents and copy map
            LocalState {
                origin: Coord2D::default(),
                map: global_state.map.clone(),
                agents: global_state.agents.clone(),
                next_agent_id: global_state.next_agent_id,
            },
            |agent_state| {
                let agent_radius = match agent_state.ty {
                    AgentType::Herbivore => AGENTS_RADIUS_HERBIVORE,
                    AgentType::Carnivore => AGENTS_RADIUS_CARNIVORE,
                };
                derive_local_state_radius(global_state, agent, MAP_RADIUS, agent_radius)
            },
        )
    }

    fn apply(global_state: &mut Self::GlobalState, local_state: &Self::State, diff: &Self::Diff) {
        // update tiles
        for (&pos, &tile) in diff.map.tiles.iter() {
            *global_state.map.at_mut(pos + local_state.origin).unwrap() = tile;
        }
        // update agents
        for (agent, mut agent_state) in diff
            .cur_agents
            .iter()
            .chain(diff.new_agents.iter())
            .cloned()
        {
            agent_state.position += local_state.origin;
            global_state.agents.insert(agent, agent_state);
        }
        // update next agent id
        if let Some(next_agent_id) = diff.next_agent_id {
            global_state.next_agent_id = next_agent_id;
        }
    }
}

impl DomainWithPlanningTask for EcosystemDomain {}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, str::FromStr};

    use super::*;
    use crate::*;

    fn create_test_global_state() -> GlobalState {
        let map = Map::from_str(
            "#0000\n\
			 01230\n\
			 ###00",
        )
        .unwrap();
        let agents = HashMap::from([
            (
                AgentId(1),
                AgentState {
                    ty: AgentType::Herbivore,
                    birth_date: 0,
                    position: Coord2D::new(1, 0),
                    food: 2,
                    death_date: None,
                },
            ),
            (
                AgentId(3),
                AgentState {
                    ty: AgentType::Carnivore,
                    birth_date: 2,
                    position: Coord2D::new(3, 2),
                    food: 5,
                    death_date: None,
                },
            ),
        ]);
        GlobalState {
            map,
            agents,
            next_agent_id: 4,
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
