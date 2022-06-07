/* 
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use crate::{AgentId, Domain, StateDiffRef, Task};

/// A possibly-recursive set of possible tasks.
pub trait Behavior<D: Domain>: 'static {
    /// Returns dependent behaviors.
    fn get_dependent_behaviors(&self) -> &'static [&'static dyn Behavior<D>] {
        &[]
    }

    /// Collects valid tasks for the given agent in the given world state.
    #[allow(unused)]
    fn add_own_tasks(&self, tick: u64, state: StateDiffRef<D>, agent: AgentId, tasks: &mut Vec<Box<dyn Task<D>>>) {}

    /// Returns if the behavior is valid for the given agent in the given world state.
    fn is_valid(&self, tick: u64, state: StateDiffRef<D>, agent: AgentId) -> bool;

    /// Helper method to recursively collect all valid tasks for the given agent in the given world state.
    fn add_tasks(&self, tick: u64, state: StateDiffRef<D>, agent: AgentId, tasks: &mut Vec<Box<dyn Task<D>>>) {
        self.add_own_tasks(tick, state, agent, tasks);
        self.get_dependent_behaviors()
            .iter()
            .filter(|behavior| behavior.is_valid(tick, state, agent))
            .for_each(|behavior| behavior.add_tasks(tick, state, agent, tasks));
    }
}
