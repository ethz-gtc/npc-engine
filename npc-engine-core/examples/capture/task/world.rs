/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use npc_engine_core::{impl_task_boxed_methods, Context, ContextMut, Task, TaskDuration};
use npc_engine_utils::OptionDiffDomain;

use crate::{
    constants::{RESPAWN_AMMO_DURATION, RESPAWN_MEDKIT_DURATION},
    domain::{CaptureDomain, DisplayAction},
    state::CapturePointState,
};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct WorldStep;

impl Task<CaptureDomain> for WorldStep {
    fn duration(&self, _ctx: Context<CaptureDomain>) -> TaskDuration {
        1
    }

    fn execute(&self, ctx: ContextMut<CaptureDomain>) -> Option<Box<dyn Task<CaptureDomain>>> {
        let diff = CaptureDomain::get_cur_state_mut(ctx.state_diff);
        // for each captured point, increment the score of the corresponding agent
        for capture_point in &diff.capture_points {
            if let CapturePointState::Captured(agent) = capture_point {
                if let Some(agent_state) = diff.agents.get_mut(agent) {
                    agent_state.acc_capture += 1;
                }
            }
        }
        // respawn if timeout
        let now = (ctx.tick & 0xff) as u8;
        if respawn_timeout_ammo(now, diff.ammo_tick) {
            diff.ammo = 1;
            diff.ammo_tick = now;
        }
        if respawn_timeout_medkit(now, diff.medkit_tick) {
            diff.medkit = 1;
            diff.medkit_tick = now;
        }

        Some(Box::new(WorldStep))
    }

    fn display_action(&self) -> DisplayAction {
        DisplayAction::WorldStep
    }

    fn is_valid(&self, _ctx: Context<CaptureDomain>) -> bool {
        true
    }

    impl_task_boxed_methods!(CaptureDomain);
}

fn respawn_timeout_ammo(now: u8, before: u8) -> bool {
    now.wrapping_sub(before) > RESPAWN_AMMO_DURATION
}
fn respawn_timeout_medkit(now: u8, before: u8) -> bool {
    now.wrapping_sub(before) > RESPAWN_MEDKIT_DURATION
}
