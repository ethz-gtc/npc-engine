/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use npc_engine_common::{AgentId, Behavior, StateDiffRef, Task};

use crate::{domain::CaptureDomain, task::world::WorldStep};

pub const WORLD_AGENT_ID: AgentId = AgentId(9);

pub struct WorldBehavior;
impl Behavior<CaptureDomain> for WorldBehavior {
    fn add_own_tasks(
        &self,
        _tick: u64,
        _state_diff: StateDiffRef<CaptureDomain>,
        _agent: AgentId,
        tasks: &mut Vec<Box<dyn Task<CaptureDomain>>>,
    ) {
        tasks.push(Box::new(WorldStep));
    }

    fn is_valid(
        &self,
        _tick: u64,
        _state_diff: StateDiffRef<CaptureDomain>,
        agent: AgentId,
    ) -> bool {
        agent == WORLD_AGENT_ID
    }
}
