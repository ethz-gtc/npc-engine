/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use npc_engine_core::{impl_task_boxed_methods, Context, ContextMut, IdleTask, Task, TaskDuration};
use npc_engine_utils::OptionDiffDomain;

use crate::{
    constants::{MAP, MAX_AMMO, MAX_HP},
    domain::{CaptureDomain, DisplayAction},
};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Pick;
impl Task<CaptureDomain> for Pick {
    fn duration(&self, _ctx: Context<CaptureDomain>) -> TaskDuration {
        // Pick is instantaneous
        0
    }

    fn weight(&self, _ctx: Context<CaptureDomain>) -> f32 {
        10.0
    }

    fn execute(&self, ctx: ContextMut<CaptureDomain>) -> Option<Box<dyn Task<CaptureDomain>>> {
        let diff = CaptureDomain::get_cur_state_mut(ctx.state_diff);
        let agent_state = diff.agents.get_mut(&ctx.agent).unwrap();
        let location = agent_state.cur_or_last_location;
        match location {
            _ if location == MAP.ammo_location() => {
                agent_state.ammo = (agent_state.ammo + 1).min(MAX_AMMO);
                diff.ammo = 0;
                diff.ammo_tick = (ctx.tick & 0xff) as u8;
            }
            _ if location == MAP.medkit_location() => {
                agent_state.hp = (agent_state.hp + 1).min(MAX_HP);
                diff.medkit = 0;
                diff.medkit_tick = (ctx.tick & 0xff) as u8;
            }
            _ => unimplemented!(),
        }
        // After Pick, the agent must wait one tick.
        Some(Box::new(IdleTask))
    }

    fn display_action(&self) -> DisplayAction {
        DisplayAction::Pick
    }

    fn is_valid(&self, ctx: Context<CaptureDomain>) -> bool {
        let state = CaptureDomain::get_cur_state(ctx.state_diff);
        state.agents.get(&ctx.agent).map_or(false, |agent_state| {
            // We cannot pick while moving.
            if agent_state.next_location.is_some() {
                false
            } else {
                // We must be at a location where there is something to pick.
                let location = agent_state.cur_or_last_location;
                (location == MAP.ammo_location() && state.ammo > 0)
                    || (location == MAP.medkit_location() && state.medkit > 0)
            }
        })
    }

    impl_task_boxed_methods!(CaptureDomain);
}
