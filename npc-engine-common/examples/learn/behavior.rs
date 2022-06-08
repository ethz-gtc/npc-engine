/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use npc_engine_common::{AgentId, Behavior, IdleTask, StateDiffRef, Task};

use crate::{
    domain::LearnDomain,
    task::{collect::Collect, left::Left, right::Right},
};

pub struct DefaultBehaviour;
impl Behavior<LearnDomain> for DefaultBehaviour {
    fn add_own_tasks(
        &self,
        tick: u64,
        state_diff: StateDiffRef<LearnDomain>,
        agent: AgentId,
        tasks: &mut Vec<Box<dyn Task<LearnDomain>>>,
    ) {
        tasks.push(Box::new(IdleTask));
        let possible_tasks: [Box<dyn Task<LearnDomain>>; 3] =
            [Box::new(Collect), Box::new(Left), Box::new(Right)];
        for task in &possible_tasks {
            if task.is_valid(tick, state_diff, agent) {
                tasks.push(task.clone());
            }
        }
    }

    fn is_valid(
        &self,
        _tick: u64,
        _state_diff: StateDiffRef<LearnDomain>,
        _agent: AgentId,
    ) -> bool {
        true
    }
}
