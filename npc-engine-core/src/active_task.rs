/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use crate::{AgentId, Context, Domain, IdleTask, Task};
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::hash::{Hash, Hasher};
use std::{fmt, mem};

/// A task associated to an agent and that is being processed by the planner.
pub struct ActiveTask<D: Domain> {
    /// the end tick of this task
    pub end: u64,
    /// the agent executing this task
    pub agent: AgentId,
    /// the actual task
    pub task: Box<dyn Task<D>>,
}
/// A set of active tasks.
///
/// These tasks are sorted by end time and then by agent, due to the implementation of `cmp` by `ActiveTask`.
pub type ActiveTasks<D> = BTreeSet<ActiveTask<D>>;

impl<D: Domain> ActiveTask<D> {
    /// Creates a new active task, computes the end from task and the state_diff.
    pub fn new(task: Box<dyn Task<D>>, ctx: Context<D>) -> Self {
        let end = ctx.tick + task.duration(ctx);
        Self::new_with_end(end, ctx.agent, task)
    }
    /// Creates a new active task with a specified end.
    pub fn new_with_end(end: u64, agent: AgentId, task: Box<dyn Task<D>>) -> Self {
        Self { end, agent, task }
    }
    /// Creates a new idle task for agent at a given tick, make sure that it will
    /// execute in the future considering that we are currently processing active_agent.
    pub fn new_idle(tick: u64, agent: AgentId, active_agent: AgentId) -> Self {
        Self {
            // Make sure the idle tasks of added agents will not be
            // executed before the active agent.
            end: if agent < active_agent { tick + 1 } else { tick },
            agent,
            task: Box::new(IdleTask),
        }
    }
    /// The memory footprint of this struct.
    pub fn size(&self, task_size: fn(&dyn Task<D>) -> usize) -> usize {
        let mut size = 0;
        size += mem::size_of::<Self>();
        size += task_size(&*self.task);
        size
    }
}

/// Returns the task associated to a given agent from an active task set.
pub(crate) fn get_task_for_agent<D: Domain>(
    set: &ActiveTasks<D>,
    agent: AgentId,
) -> Option<&ActiveTask<D>> {
    for task in set {
        if task.agent == agent {
            return Some(task);
        }
    }
    None
}

impl<D: Domain> fmt::Debug for ActiveTask<D> {
    fn fmt(&self, f: &'_ mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ActiveTask")
            .field("end", &self.end)
            .field("agent", &self.agent)
            .field("task", &self.task)
            .finish()
    }
}

impl<D: Domain> std::fmt::Display for ActiveTask<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {:?} ends T{}", self.agent, self.task, self.end)
    }
}

impl<D: Domain> Hash for ActiveTask<D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.end.hash(state);
        self.agent.hash(state);
        self.task.hash(state);
    }
}

impl<D: Domain> Clone for ActiveTask<D> {
    fn clone(&self) -> Self {
        ActiveTask {
            end: self.end,
            agent: self.agent,
            task: self.task.box_clone(),
        }
    }
}

impl<D: Domain> Ord for ActiveTask<D> {
    /// Active tasks are ordered first by time of ending, then by agent id
    fn cmp(&self, other: &Self) -> Ordering {
        self.end.cmp(&other.end).then(self.agent.cmp(&other.agent))
    }
}

impl<D: Domain> PartialOrd for ActiveTask<D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<D: Domain> Eq for ActiveTask<D> {}

impl<D: Domain> PartialEq for ActiveTask<D> {
    fn eq(&self, other: &Self) -> bool {
        self.end == other.end && self.agent == other.agent && self.task.box_eq(&other.task)
    }
}
