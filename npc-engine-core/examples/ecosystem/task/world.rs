/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::collections::HashSet;

use npc_engine_core::{
    impl_task_boxed_methods, AgentId, StateDiffRef, StateDiffRefMut, Task, TaskDuration,
};

use crate::{
    constants::WORLD_TASK_DURATION,
    domain::{DisplayAction, EcosystemDomain},
    state::AgentState,
};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct WorldStep;

impl Task<EcosystemDomain> for WorldStep {
    fn duration(
        &self,
        _tick: u64,
        _state_diff: StateDiffRef<EcosystemDomain>,
        _agent: AgentId,
    ) -> TaskDuration {
        WORLD_TASK_DURATION
    }

    fn execute(
        &self,
        _tick: u64,
        state_diff: StateDiffRefMut<EcosystemDomain>,
        _agent: AgentId,
    ) -> Option<Box<dyn Task<EcosystemDomain>>> {
        let age_agent = |state: &mut AgentState| {
            if state.food > 0 {
                state.food -= 1;
                if state.food == 0 {
                    state.alive = false;
                }
            }
        };
        // process agents from the diff
        let mut seen_agents = HashSet::new();
        for (agent, agent_state) in state_diff.diff.agents.iter_mut() {
            seen_agents.insert(*agent);
            age_agent(agent_state);
        }
        // then from the state
        for (agent, agent_state) in state_diff.initial_state.agents.iter() {
            if !seen_agents.contains(agent) {
                let mut agent_state = agent_state.clone();
                age_agent(&mut agent_state);
                state_diff.diff.agents.push((*agent, agent_state.clone()));
            }
        }

        Some(Box::new(WorldStep))
    }

    fn display_action(&self) -> DisplayAction {
        DisplayAction::WorldStep
    }

    fn is_valid(
        &self,
        _tick: u64,
        _state_diff: StateDiffRef<EcosystemDomain>,
        _agent: AgentId,
    ) -> bool {
        true
    }

    impl_task_boxed_methods!(EcosystemDomain);
}
