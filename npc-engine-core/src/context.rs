/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use crate::{AgentId, Domain, StateDiffRef, StateDiffRefMut};

/// The context of a search node
pub struct Context<'a, D: Domain> {
    /// tick at this node
    pub tick: u64,
    /// world state at this node
    pub state_diff: StateDiffRef<'a, D>,
    /// agent for this node
    pub agent: AgentId,
}
impl<D: Domain> Copy for Context<'_, D> {}
impl<D: Domain> Clone for Context<'_, D> {
    fn clone(&self) -> Self {
        Self {
            tick: self.tick,
            state_diff: self.state_diff,
            agent: self.agent,
        }
    }
}
impl<'a, D: Domain> Context<'a, D> {
    /// Creates a new Context from its components.
    pub fn new(tick: u64, state_diff: StateDiffRef<'a, D>, agent: AgentId) -> Self {
        Self {
            tick,
            state_diff,
            agent,
        }
    }
    /// Builds directly from an initial_state and diff .
    pub fn with_state_and_diff(
        tick: u64,
        initial_state: &'a D::State,
        diff: &'a D::Diff,
        agent: AgentId,
    ) -> Self {
        Self {
            tick,
            state_diff: StateDiffRef::new(initial_state, diff),
            agent,
        }
    }
    /// Replaces the tick and agent, keep the state_diff.
    pub fn replace_tick_and_agent(self, tick: u64, agent: AgentId) -> Self {
        Self {
            tick,
            state_diff: self.state_diff,
            agent,
        }
    }
    /// Drops the state_diff and returns (tick, agent)
    pub fn drop_state_diff(self) -> (u64, AgentId) {
        (self.tick, self.agent)
    }
}

/// The context of a search node, mutable version
pub struct ContextMut<'a, D: Domain> {
    /// tick at this node
    pub tick: u64,
    /// world state at this node
    pub state_diff: StateDiffRefMut<'a, D>,
    /// agent for this node
    pub agent: AgentId,
}
impl<'a, D: Domain> ContextMut<'a, D> {
    /// Creates a new ContextMut from its components.
    pub fn new(tick: u64, state_diff: StateDiffRefMut<'a, D>, agent: AgentId) -> Self {
        Self {
            tick,
            state_diff,
            agent,
        }
    }
    /// Builds directly from an initial_state and diff.
    pub fn with_state_and_diff(
        tick: u64,
        initial_state: &'a D::State,
        diff: &'a mut D::Diff,
        agent: AgentId,
    ) -> Self {
        Self {
            tick,
            state_diff: StateDiffRefMut::new(initial_state, diff),
            agent,
        }
    }
    /// Builds directly from an initial_state and diff, with rest as a tuple.
    pub fn with_rest_and_state_and_diff(
        rest: (u64, AgentId),
        initial_state: &'a D::State,
        diff: &'a mut D::Diff,
    ) -> Self {
        Self {
            tick: rest.0,
            state_diff: StateDiffRefMut::new(initial_state, diff),
            agent: rest.1,
        }
    }
}
