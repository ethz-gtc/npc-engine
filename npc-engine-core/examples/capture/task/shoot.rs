/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use npc_engine_core::{
    impl_task_boxed_methods, AgentId, Context, ContextMut, IdleTask, Task, TaskDuration,
};
use npc_engine_utils::OptionDiffDomain;

use crate::domain::{CaptureDomain, DisplayAction};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Shoot(pub AgentId);
impl Task<CaptureDomain> for Shoot {
    fn duration(&self, _ctx: Context<CaptureDomain>) -> TaskDuration {
        // Shoot is instantaneous
        0
    }

    fn weight(&self, _ctx: Context<CaptureDomain>) -> f32 {
        10.0
    }

    fn execute(&self, ctx: ContextMut<CaptureDomain>) -> Option<Box<dyn Task<CaptureDomain>>> {
        let diff = CaptureDomain::get_cur_state_mut(ctx.state_diff);
        let agent_state = diff.agents.get_mut(&ctx.agent).unwrap();
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

    fn is_valid(&self, ctx: Context<CaptureDomain>) -> bool {
        let state = CaptureDomain::get_cur_state(ctx.state_diff);
        state.agents.get(&ctx.agent).map_or(false, |agent_state| {
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
