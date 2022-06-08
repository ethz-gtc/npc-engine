/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use npc_engine_common::{
    impl_task_boxed_methods, AgentId, StateDiffRef, StateDiffRefMut, Task, TaskDuration,
};

use crate::{
    constants::*,
    domain::{DisplayAction, EcosystemDomain},
    map::Tile,
    state::{Access, AccessMut},
};

#[derive(Hash, Clone, Eq, PartialEq, Debug)]
pub struct EatGrass;

impl Task<EcosystemDomain> for EatGrass {
    fn weight(
        &self,
        _tick: u64,
        _state_diff: StateDiffRef<EcosystemDomain>,
        _agent: AgentId,
    ) -> f32 {
        EAT_GRASS_WEIGHT
    }

    fn duration(
        &self,
        _tick: u64,
        _state_diff: StateDiffRef<EcosystemDomain>,
        _agent: AgentId,
    ) -> TaskDuration {
        0
    }

    fn execute(
        &self,
        _tick: u64,
        mut state_diff: StateDiffRefMut<EcosystemDomain>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<EcosystemDomain>>> {
        let agent_state = state_diff.get_agent_mut(agent).unwrap();
        agent_state.food = HERBIVORE_MAX_FOOD;
        let agent_pos = agent_state.position;
        let growth = state_diff.get_grass(agent_pos).unwrap();
        state_diff.set_tile(agent_pos, Tile::Grass(growth - 1));
        None
    }

    fn is_valid(
        &self,
        _tick: u64,
        state_diff: StateDiffRef<EcosystemDomain>,
        agent: AgentId,
    ) -> bool {
        let agent_state = state_diff.get_agent(agent).unwrap();
        debug_assert!(
            agent_state.alive,
            "Task validity check called on a dead agent"
        );
        if !agent_state.alive {
            return false;
        }
        if !agent_state.food < HERBIVORE_MAX_FOOD {
            return false;
        }
        state_diff
            .get_grass(agent_state.position)
            .filter(|growth| *growth > 0)
            .is_some()
    }

    fn display_action(&self) -> DisplayAction {
        DisplayAction::EatGrass
    }

    impl_task_boxed_methods!(EcosystemDomain);
}
