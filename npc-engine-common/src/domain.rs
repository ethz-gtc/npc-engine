use std::collections::BTreeSet;
use std::hash::Hash;
use std::ops::Deref;
use std::{fmt, mem};

use crate::{AgentId, Behavior, Task};

// TODO: remove debug constraints
/// A domain on which the MCTS planner can plan
pub trait Domain: Sized + 'static {
    /// The state the MCTS plans on.
    type State: std::fmt::Debug + Sized + 'static;
    /// A compact set of changes towards a `State` that are accumulated throughout planning.
    type Diff: std::fmt::Debug + Default + Clone + Hash + Eq + 'static;
    /// A representation of a display action that can be fetched from a task.
    type DisplayAction;

    /// Returns all behaviors available for this domain.
    fn list_behaviors() -> &'static [&'static dyn Behavior<Self>];

    /// Gets the current value of the given agent in the given world state.
    fn get_current_value(state_diff: StateDiffRef<Self>, agent: AgentId) -> f32;

    /// Updates the list of agents which are in the horizon of the given agent in the given world state.
    fn update_visible_agents(state_diff: StateDiffRef<Self>, agent: AgentId, agents: &mut BTreeSet<AgentId>);

    /// Gets all agents which are in the horizon of the given agent in the given world state.
    fn get_visible_agents(state_diff: StateDiffRef<Self>, agent: AgentId) -> BTreeSet<AgentId> {
        let mut agents = BTreeSet::new();
        Self::update_visible_agents(state_diff, agent, &mut agents);
        agents
    }

    /// Gets all possible valid tasks for a given agent in a given world state.
    fn get_tasks<'a>(
        state_diff: StateDiffRef<'a, Self>,
        agent: AgentId
    ) -> Vec<Box<dyn Task<Self>>> {
        let mut actions = Vec::new();
        Self::list_behaviors()
            .iter()
            .filter(|behavior| behavior.is_valid(state_diff, agent))
            .for_each(|behavior| behavior.add_tasks(state_diff, agent, &mut actions));

        actions.dedup();
        actions
    }
}

pub struct StateDiffRef<'a, D: Domain> {
    pub initial_state: &'a D::State, // FIXME: unpub
    pub diff: &'a D::Diff, // FIXME: unpub
}
impl<D: Domain> Copy for StateDiffRef<'_, D> {}
impl<D: Domain> Clone for StateDiffRef<'_, D> {
    fn clone(&self) -> Self {
        StateDiffRef::new(self.initial_state, self.diff)
    }
}
impl<'a, D: Domain> StateDiffRef<'a, D> {
    pub fn new(initial_state: &'a D::State, diff: &'a D::Diff) -> Self {
        StateDiffRef { initial_state, diff }
    }
}

impl<D: Domain> fmt::Debug for StateDiffRef<'_, D>
where
    D::State: fmt::Debug,
    D::Diff: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f
            .debug_struct("SnapshotDiffRef")
            .field("Snapshot", self.initial_state)
            .field("Diff", self.diff)
            .finish()
    }
}

pub struct StateDiffRefMut<'a, D: Domain> {
    pub initial_state: &'a D::State, // FIXME: unpub
    pub diff: &'a mut D::Diff, // FIXME: unpub
}
impl<'a, D: Domain> StateDiffRefMut<'a, D> {
    pub fn new(initial_state: &'a D::State, diff: &'a mut D::Diff) -> Self {
        StateDiffRefMut { initial_state, diff }
    }
}

impl<'a, D: Domain> Deref for StateDiffRefMut<'a, D> {
    type Target = StateDiffRef<'a, D>;

    fn deref(&self) -> &Self::Target {
        // Safety: StateDiffRef and StateDiffRefMut have the same memory layout
        // and casting from mutable to immutable is always safe
        unsafe { mem::transmute(self) }
    }
}