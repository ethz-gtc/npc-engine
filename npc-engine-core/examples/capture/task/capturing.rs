/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use npc_engine_core::{impl_task_boxed_methods, Context, ContextMut, Task, TaskDuration};
use npc_engine_utils::OptionDiffDomain;

use crate::{
    constants::{CAPTURE_DURATION, MAP},
    domain::{CaptureDomain, DisplayAction},
    state::CapturePointState,
};

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct StartCapturing(pub u8);
impl Task<CaptureDomain> for StartCapturing {
    fn duration(&self, _ctx: Context<CaptureDomain>) -> TaskDuration {
        // StartCapture is instantaneous
        0
    }

    fn execute(&self, ctx: ContextMut<CaptureDomain>) -> Option<Box<dyn Task<CaptureDomain>>> {
        let diff = CaptureDomain::get_cur_state_mut(ctx.state_diff);
        diff.capture_points[self.0 as usize] = CapturePointState::Capturing(ctx.agent);
        Some(Box::new(Capturing(self.0)))
    }

    fn display_action(&self) -> DisplayAction {
        DisplayAction::StartCapturing(MAP.capture_location(self.0))
    }

    fn is_valid(&self, ctx: Context<CaptureDomain>) -> bool {
        let state = CaptureDomain::get_cur_state(ctx.state_diff);
        // if the point is already captured, we cannot restart capturing
        if state.capture_points[self.0 as usize] == CapturePointState::Captured(ctx.agent) {
            return false;
        }
        let capture_location = MAP.capture_location(self.0);
        state.agents.get(&ctx.agent).map_or(false, |agent_state|
				// agent is at the right location and not moving
				agent_state.cur_or_last_location == capture_location &&
				agent_state.next_location.is_none())
    }

    impl_task_boxed_methods!(CaptureDomain);
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Capturing(u8);
impl Task<CaptureDomain> for Capturing {
    fn duration(&self, _ctx: Context<CaptureDomain>) -> TaskDuration {
        // Capturing takes some time
        CAPTURE_DURATION
    }

    fn execute(&self, ctx: ContextMut<CaptureDomain>) -> Option<Box<dyn Task<CaptureDomain>>> {
        let diff = CaptureDomain::get_cur_state_mut(ctx.state_diff);
        diff.capture_points[self.0 as usize] = CapturePointState::Captured(ctx.agent);
        None
    }

    fn display_action(&self) -> DisplayAction {
        DisplayAction::Capturing(MAP.capture_location(self.0))
    }

    fn is_valid(&self, ctx: Context<CaptureDomain>) -> bool {
        let state = CaptureDomain::get_cur_state(ctx.state_diff);
        state.agents.get(&ctx.agent).is_some()
            && state.capture_points[self.0 as usize] == CapturePointState::Capturing(ctx.agent)
        // note: no need to check agent location, as this task is always a follow-up of StartCapturing
    }

    impl_task_boxed_methods!(CaptureDomain);
}
