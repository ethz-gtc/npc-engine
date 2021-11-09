use std::fmt;

use crate::{AgentId, NpcEngine, StateRef, Task};

pub trait Behavior<E: NpcEngine>: fmt::Display + 'static {
    /// Returns dependent behaviors.
    fn behaviors(&self) -> &'static [&'static dyn Behavior<E>] {
        &[]
    }

    /// Sets list of tasks for the given `state` and `agent`.
    #[allow(unused)]
    fn tasks(&self, state: StateRef<E>, agent: AgentId, tasks: &mut Vec<Box<dyn Task<E>>>) {}

    /// Returns `true` if the behavior is valid for the given state and agent
    fn predicate(&self, state: StateRef<E>, agent: AgentId) -> bool;

    /// Helper method to recursively add all tasks
    fn add_tasks(&self, state: StateRef<E>, agent: AgentId, tasks: &mut Vec<Box<dyn Task<E>>>) {
        self.tasks(state, agent, tasks);
        self.behaviors()
            .iter()
            .filter(|behavior| behavior.predicate(state, agent))
            .for_each(|behavior| behavior.add_tasks(state, agent, tasks));
    }
}
