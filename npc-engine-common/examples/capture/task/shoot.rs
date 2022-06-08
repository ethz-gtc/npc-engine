/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use npc_engine_common::{
    impl_task_boxed_methods, AgentId, IdleTask, StateDiffRef, StateDiffRefMut, Task, TaskDuration,
};
use npc_engine_utils::OptionDiffDomain;

use crate::domain::{CaptureDomain, DisplayAction};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Shoot(pub AgentId);
impl Task<CaptureDomain> for Shoot {
    fn duration(
        &self,
        _tick: u64,
        _state_diff: StateDiffRef<CaptureDomain>,
        _agent: AgentId,
    ) -> TaskDuration {
        // Shoot is instantaneous
        0
    }

    fn weight(&self, _tick: u64, _state_diff: StateDiffRef<CaptureDomain>, _agent: AgentId) -> f32 {
        10.0
    }

    fn execute(
        &self,
        _tick: u64,
        state_diff: StateDiffRefMut<CaptureDomain>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<CaptureDomain>>> {
        let diff = CaptureDomain::get_cur_state_mut(state_diff);
        let agent_state = diff.agents.get_mut(&agent).unwrap();
        agent_state.ammo -= 1;
        let target_state = diff.agents.get_mut(&self.0).unwrap();
        if target_state.hp > 0 {
            target_state.hp -= 1;
        }
        if target_state.hp == 0 {
            diff.agents.remove(&self.0);
        }
        // After Shoot, the agent must wait one tick.
        Some(Box::new(IdleTask))
    }

    fn display_action(&self) -> DisplayAction {
        DisplayAction::Shoot(self.0)
    }

    fn is_valid(
        &self,
        _tick: u64,
        state_diff: StateDiffRef<CaptureDomain>,
        agent: AgentId,
    ) -> bool {
        let state = CaptureDomain::get_cur_state(state_diff);
        state.agents.get(&agent).map_or(false, |agent_state| {
            // We must have ammo to shoot.
            // We cannot shoot while moving.
            if agent_state.ammo == 0 || agent_state.next_location.is_some() {
                false
            } else {
                let location = agent_state.cur_or_last_location;
                // Target must exist.
                let target = state.agents.get(&self.0);
                target.map_or(false, |target| {
                    // Target must be on our location or its adjacent paths.
                    target.cur_or_last_location == location
                        || target
                            .next_location
                            .map_or(false, |next_location| next_location == location)
                })
            }
        })
    }

    impl_task_boxed_methods!(CaptureDomain);
}
