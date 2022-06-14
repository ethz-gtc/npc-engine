/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::{collections::BTreeMap, fmt};

use npc_engine_core::AgentId;

use crate::map::Location;

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct AgentState {
    /// accumulated capture points for this agent
    pub acc_capture: u16,
    /// current or last location of the agent (if travelling),
    pub cur_or_last_location: Location,
    /// next location of the agent, if travelling, none otherwise
    pub next_location: Option<Location>,
    /// health point of the agent
    pub hp: u8,
    /// ammunition carried by the agent
    pub ammo: u8,
}

#[derive(Clone, Hash, PartialEq, Eq)]
pub enum CapturePointState {
    Free,
    Capturing(AgentId),
    Captured(AgentId),
}
impl fmt::Debug for CapturePointState {
    fn fmt(&self, f: &'_ mut fmt::Formatter) -> fmt::Result {
        match &self {
            CapturePointState::Free => f.write_str("__"),
            CapturePointState::Capturing(agent) => f.write_fmt(format_args!("C{:?}", agent.0)),
            CapturePointState::Captured(agent) => f.write_fmt(format_args!("H{:?}", agent.0)),
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct State {
    /// active agents
    pub agents: BTreeMap<AgentId, AgentState>,
    /// capture points
    pub capture_points: [CapturePointState; 3],
    /// ammo available at collection point
    pub ammo: u8,
    /// tick when ammo was collected
    pub ammo_tick: u8,
    /// medical kit available at collection point
    pub medkit: u8,
    /// tick when med kit was collected
    pub medkit_tick: u8,
}

pub type Diff = Option<State>; // if Some, use this diff, otherwise use initial state
