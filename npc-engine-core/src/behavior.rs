/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use crate::{AgentId, Domain, StateDiffRef, Task};

/// A possibly-recursive set of possible tasks.
///
/// You need to implement at least two methods: [is_valid](Self::is_valid) and  [add_own_tasks](Self::add_own_tasks).
pub trait Behavior<D: Domain>: 'static {
    /// Returns if the behavior is valid for the given agent in the given world state.
    fn is_valid(&self, tick: u64, state: StateDiffRef<D>, agent: AgentId) -> bool;

    /// Collects valid tasks for the given agent in the given world state.
    #[allow(unused)]
    fn add_own_tasks(
        &self,
        tick: u64,
        state: StateDiffRef<D>,
        agent: AgentId,
        tasks: &mut Vec<Box<dyn Task<D>>>,
    );

    /// Returns dependent behaviors.
    fn get_dependent_behaviors(&self) -> &'static [&'static dyn Behavior<D>] {
        &[]
    }

    /// Helper method to recursively collect all valid tasks for the given agent in the given world state.
    ///
    /// It will not do anything if the behavior is invalid at that point.
    fn add_tasks(
        &self,
        tick: u64,
        state: StateDiffRef<D>,
        agent: AgentId,
        tasks: &mut Vec<Box<dyn Task<D>>>,
    ) {
        if !self.is_valid(tick, state, agent) {
            return;
        }
        self.add_own_tasks(tick, state, agent, tasks);
        self.get_dependent_behaviors()
            .iter()
            .for_each(|behavior| behavior.add_tasks(tick, state, agent, tasks));
    }
}
