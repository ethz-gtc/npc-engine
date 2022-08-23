/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use npc_engine_core::{AgentId, AgentValue, Behavior, Context, Domain, StateDiffRef};
use npc_engine_utils::OptionDiffDomain;

use crate::{behavior::DefaultBehaviour, state::State};

pub type Diff = Option<State>; // if Some, use this diff, otherwise use initial state

#[derive(Debug)]
pub enum DisplayAction {
    Wait,
    Collect,
    Left,
    Right,
}

impl Default for DisplayAction {
    fn default() -> Self {
        Self::Wait
    }
}

pub struct LearnDomain;

impl Domain for LearnDomain {
    type State = State;
    type Diff = Diff;
    type DisplayAction = DisplayAction;

    fn list_behaviors() -> &'static [&'static dyn Behavior<Self>] {
        &[&DefaultBehaviour]
    }

    fn get_current_value(
        _tick: u64,
        state_diff: StateDiffRef<Self>,
        _agent: AgentId,
    ) -> AgentValue {
        let state = Self::get_cur_state(state_diff);
        AgentValue::from(state.wood_count)
    }

    fn update_visible_agents(
        _start_tick: u64,
        ctx: Context<Self>,
        agents: &mut std::collections::BTreeSet<AgentId>,
    ) {
        agents.insert(ctx.agent);
    }

    fn get_state_description(state_diff: StateDiffRef<Self>) -> String {
        let state = Self::get_cur_state(state_diff);
        let map = state
            .map
            .iter()
            .map(|count| match count {
                0 => "  ",
                1 => "🌱",
                2 => "🌿",
                _ => "🌳",
            })
            .collect::<String>();
        format!("@{:2} 🪵{:2} [{map}]", state.agent_pos, state.wood_count)
    }
}
