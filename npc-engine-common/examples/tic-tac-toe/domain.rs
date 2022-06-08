/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::{collections::BTreeSet, fmt};

use npc_engine_common::{AgentId, AgentValue, Behavior, Domain, StateDiffRef};

use crate::{
    board::{winner, CellArray2D, Diff, State},
    r#move::{Move, MoveBehavior},
};

// Option, so that the idle placeholder action is Wait
#[derive(Default)]
pub struct DisplayAction(pub Option<Move>);

impl fmt::Debug for DisplayAction {
    fn fmt(&self, f: &'_ mut fmt::Formatter) -> fmt::Result {
        match &self.0 {
            None => f.write_str("Wait"),
            Some(m) => f.write_fmt(format_args!("Move({}, {})", m.x, m.y)),
        }
    }
}

pub struct TicTacToe;

// TODO: once const NotNan::new() is stabilized, switch to that
// SAFETY: 0.0, 1.0, -1.0 are not NaN
const VALUE_UNDECIDED: AgentValue = unsafe { AgentValue::new_unchecked(0.) };
const VALUE_WIN: AgentValue = unsafe { AgentValue::new_unchecked(1.) };
const VALUE_LOOSE: AgentValue = unsafe { AgentValue::new_unchecked(-1.) };

impl Domain for TicTacToe {
    type State = State;
    type Diff = Diff;
    type DisplayAction = DisplayAction;

    fn list_behaviors() -> &'static [&'static dyn Behavior<Self>] {
        &[&MoveBehavior]
    }

    fn get_current_value(_tick: u64, state_diff: StateDiffRef<Self>, agent: AgentId) -> AgentValue {
        let state = *state_diff.initial_state | state_diff.diff.unwrap_or(0);
        match winner(state) {
            None => VALUE_UNDECIDED,
            Some(player) => {
                if player.to_agent() == agent {
                    VALUE_WIN
                } else {
                    VALUE_LOOSE
                }
            }
        }
    }

    fn update_visible_agents(
        _start_tick: u64,
        _tick: u64,
        _state_diff: StateDiffRef<Self>,
        _agent: AgentId,
        agents: &mut BTreeSet<AgentId>,
    ) {
        agents.insert(AgentId(0));
        agents.insert(AgentId(1));
    }

    fn get_state_description(state_diff: StateDiffRef<Self>) -> String {
        let state = *state_diff.initial_state | state_diff.diff.unwrap_or(0);
        state.description()
    }
}
