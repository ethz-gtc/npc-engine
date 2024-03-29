/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use npc_engine_core::{AgentId, Behavior, Context, IdleTask, Task};
use npc_engine_utils::OptionDiffDomain;

use crate::{
    constants::MAP,
    domain::CaptureDomain,
    task::{capturing::StartCapturing, moving::StartMoving, pick::Pick, shoot::Shoot},
};

use super::world::WORLD_AGENT_ID;

pub struct AgentBehavior;
impl Behavior<CaptureDomain> for AgentBehavior {
    fn add_own_tasks(
        &self,
        ctx: Context<CaptureDomain>,
        tasks: &mut Vec<Box<dyn Task<CaptureDomain>>>,
    ) {
        let state = CaptureDomain::get_cur_state(ctx.state_diff);
        let agent_state = state.agents.get(&ctx.agent);
        if let Some(agent_state) = agent_state {
            // already moving, cannot do anything else
            if agent_state.next_location.is_some() {
                return;
            }
            tasks.push(Box::new(IdleTask));
            for to in MAP.neighbors(agent_state.cur_or_last_location) {
                let task = StartMoving { to };
                tasks.push(Box::new(task));
            }
            let other_agent = if ctx.agent.0 == 0 {
                AgentId(1)
            } else {
                AgentId(0)
            };
            let other_tasks: Vec<Box<dyn Task<CaptureDomain>>> =
                vec![Box::new(Pick), Box::new(Shoot(other_agent))];
            for task in other_tasks {
                if task.is_valid(ctx) {
                    tasks.push(task);
                }
            }
            for capture_index in 0..MAP.capture_locations_count() {
                let task = StartCapturing(capture_index);
                if task.is_valid(ctx) {
                    tasks.push(Box::new(task));
                }
            }
        }
    }

    fn is_valid(&self, ctx: Context<CaptureDomain>) -> bool {
        ctx.agent != WORLD_AGENT_ID
    }
}
