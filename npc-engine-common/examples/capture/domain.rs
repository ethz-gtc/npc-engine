/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::{collections::BTreeSet, fmt};

use npc_engine_common::{AgentId, AgentValue, Behavior, Domain, StateDiffRef};
use npc_engine_utils::OptionDiffDomain;
use num_traits::Zero;

use crate::{
    behavior::{
        agent::AgentBehavior,
        world::{WorldBehavior, WORLD_AGENT_ID},
    },
    constants::MAP,
    map::Location,
    state::{Diff, State},
};

pub enum DisplayAction {
    Wait,
    Pick,
    Shoot(AgentId),
    StartCapturing(Location),
    Capturing(Location),
    StartMoving(Location),
    Moving(Location),
    WorldStep,
}

impl Default for DisplayAction {
    fn default() -> Self {
        Self::Wait
    }
}

impl fmt::Debug for DisplayAction {
    fn fmt(&self, f: &'_ mut fmt::Formatter) -> fmt::Result {
        match &self {
            Self::Wait => f.write_str("Wait"),
            Self::Pick => f.write_str("Pick"),
            Self::Shoot(target) => f.write_fmt(format_args!("Shoot {:?}", target)),
            Self::StartCapturing(loc) => f.write_fmt(format_args!("StartCapturing {:?}", loc)),
            Self::Capturing(loc) => f.write_fmt(format_args!("Capturing {:?}", loc)),
            Self::StartMoving(loc) => f.write_fmt(format_args!("StartMoving {:?}", loc)),
            Self::Moving(loc) => f.write_fmt(format_args!("Moving {:?}", loc)),
            Self::WorldStep => f.write_str("WorldStep"),
        }
    }
}

pub struct CaptureDomain;

impl Domain for CaptureDomain {
    type State = State;
    type Diff = Diff;
    type DisplayAction = DisplayAction;

    fn list_behaviors() -> &'static [&'static dyn Behavior<Self>] {
        &[&AgentBehavior, &WorldBehavior]
    }

    fn get_current_value(_tick: u64, state_diff: StateDiffRef<Self>, agent: AgentId) -> AgentValue {
        let state = Self::get_cur_state(state_diff);
        state
            .agents
            .get(&agent)
            .map_or(AgentValue::zero(), |agent_state| {
                AgentValue::from(agent_state.acc_capture)
            })
    }

    fn update_visible_agents(
        _start_tick: u64,
        _tick: u64,
        state_diff: StateDiffRef<Self>,
        _agent: AgentId,
        agents: &mut BTreeSet<AgentId>,
    ) {
        let state = Self::get_cur_state(state_diff);
        agents.clear();
        agents.extend(state.agents.keys());
        agents.insert(WORLD_AGENT_ID);
    }

    fn get_state_description(state_diff: StateDiffRef<Self>) -> String {
        let state = Self::get_cur_state(state_diff);
        let mut s = format!(
            "World: ❤️ {} ({}), • {} ({}), ⚡: ",
            state.medkit, state.medkit_tick, state.ammo, state.ammo_tick
        );
        s += &(0..MAP.capture_locations_count())
            .map(|index| format!("{:?}", state.capture_points[index as usize]))
            .collect::<Vec<_>>()
            .join(" ");
        for (id, state) in &state.agents {
            if let Some(target) = state.next_location {
                s += &format!(
                    "\nA{} in {}-{}, ❤️ {}, • {}, ⚡{}",
                    id.0,
                    state.cur_or_last_location.get(),
                    target.get(),
                    state.hp,
                    state.ammo,
                    state.acc_capture
                );
            } else {
                s += &format!(
                    "\nA{} @    {}, ❤️ {}, • {}, ⚡{}",
                    id.0,
                    state.cur_or_last_location.get(),
                    state.hp,
                    state.ammo,
                    state.acc_capture
                );
            }
        }
        s
    }
}
