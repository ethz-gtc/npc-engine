/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

// use std::fmt::{self, Formatter};

use npc_engine_core::{impl_task_boxed_methods, Context, ContextMut, Task, TaskDuration};
use npc_engine_utils::Direction;

use crate::{
    constants::MOVE_WEIGHT,
    domain::{DisplayAction, EcosystemDomain},
    map::DirConv,
    state::{Access, AccessMut},
};

#[derive(Hash, Clone, Eq, PartialEq, Debug)]
pub struct Move(pub Direction);

impl Task<EcosystemDomain> for Move {
    fn weight(&self, _ctx: Context<EcosystemDomain>) -> f32 {
        MOVE_WEIGHT
    }

    fn duration(&self, _ctx: Context<EcosystemDomain>) -> TaskDuration {
        0
    }

    fn execute(
        &self,
        mut ctx: ContextMut<EcosystemDomain>,
    ) -> Option<Box<dyn Task<EcosystemDomain>>> {
        let agent_state = ctx.state_diff.get_agent_mut(ctx.agent).unwrap();
        let target_pos = DirConv::apply(self.0, agent_state.position);
        agent_state.position = target_pos;
        None
    }

    fn is_valid(&self, ctx: Context<EcosystemDomain>) -> bool {
        let agent_state = ctx.state_diff.get_agent(ctx.agent).unwrap();
        debug_assert!(
            agent_state.alive(),
            "Task validity check called on a dead agent"
        );
        if !agent_state.alive() {
            return false;
        }
        let target_pos = DirConv::apply(self.0, agent_state.position);
        ctx.state_diff.is_position_free(target_pos)
    }

    fn display_action(&self) -> DisplayAction {
        DisplayAction::Move(self.0)
    }

    impl_task_boxed_methods!(EcosystemDomain);
}
