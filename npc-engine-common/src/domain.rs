use std::collections::BTreeSet;
use std::hash::Hash;
use std::ops::Deref;
use std::{fmt, mem};

use crate::{AgentId, Behavior, Task};

// TODO: remove debug constraints
/// A domain on which the MCTS planner can plan
pub trait Domain: Sized + 'static {
    /// World state: all data that can change in the course of the simulation.
    type State: std::fmt::Debug + Sized + 'static;
    /// Possibly a smaller view of the `State`, as seen by a give agent.
    type Snapshot: std::fmt::Debug + Sized + 'static;
    /// A compact set of changes towards a `Snapshot` that are accumulated throughout planning.
    type Diff: std::fmt::Debug + Default + Clone + Hash + Eq + 'static;
    /// A representation of a display action that can be fetched from a task.
    type DisplayAction;

    /// Returns all behaviors available for this domain.
    fn list_behaviors() -> &'static [&'static dyn Behavior<Self>];

    /// Derives a new `Snapshot` for the given agent from the given world state.
    fn derive_snapshot(state: &Self::State, agent: AgentId) -> Self::Snapshot;

    /// Applies a diff from a snapshot to the world state.
    fn apply(state: &mut Self::State, snapshot: &Self::Snapshot, diff: &Self::Diff);

    /// Gets the current value of the given agent in the given world state.
    fn get_current_value(state: SnapshotDiffRef<Self>, agent: AgentId) -> f32;

    /// Updates the list of agents which are in the horizon of the given agent in the given world state.
    fn update_visible_agents(state: SnapshotDiffRef<Self>, agent: AgentId, agents: &mut BTreeSet<AgentId>);

    /// Gets all agents which are in the horizon of the given agent in the given world state.
    fn get_visible_agents(state: SnapshotDiffRef<Self>, agent: AgentId) -> BTreeSet<AgentId> {
        let mut agents = BTreeSet::new();
        Self::update_visible_agents(state, agent, &mut agents);
        agents
    }

    /// Gets all possible valid tasks for a given agent in a given world state.
    fn get_tasks<'a>(
        state: SnapshotDiffRef<'a, Self>,
        agent: AgentId
    ) -> Vec<Box<dyn Task<Self>>> {
        let mut actions = Vec::new();
        Self::list_behaviors()
            .iter()
            .filter(|behavior| behavior.is_valid(state, agent))
            .for_each(|behavior| behavior.add_tasks(state, agent, &mut actions));

        actions.dedup();
        actions
    }
}

pub struct SnapshotDiffRef<'a, D: Domain> {
    pub snapshot: &'a D::Snapshot, // FIXME: unpub
    pub diff: &'a D::Diff, // FIXME: unpub
}
impl<D: Domain> Copy for SnapshotDiffRef<'_, D> {}
impl<D: Domain> Clone for SnapshotDiffRef<'_, D> {
    fn clone(&self) -> Self {
        SnapshotDiffRef::new(self.snapshot, self.diff)
    }
}
impl<'a, D: Domain> SnapshotDiffRef<'a, D> {
    pub fn new(snapshot: &'a D::Snapshot, diff: &'a D::Diff) -> Self {
        SnapshotDiffRef { snapshot, diff }
    }
}

impl<D: Domain> fmt::Debug for SnapshotDiffRef<'_, D>
where
    D::State: fmt::Debug,
    D::Snapshot: fmt::Debug,
    D::Diff: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f
            .debug_struct("SnapshotDiffRef")
            .field("Snapshot", self.snapshot)
            .field("Diff", self.diff)
            .finish()
    }
}

pub struct SnapshotDiffRefMut<'a, D: Domain> {
    pub snapshot: &'a D::Snapshot, // FIXME: unpub
    pub diff: &'a mut D::Diff, // FIXME: unpub
}
impl<'a, D: Domain> SnapshotDiffRefMut<'a, D> {
    pub fn new(snapshot: &'a D::Snapshot, diff: &'a mut D::Diff) -> Self {
        SnapshotDiffRefMut { snapshot, diff }
    }
}

impl<'a, D: Domain> Deref for SnapshotDiffRefMut<'a, D> {
    type Target = SnapshotDiffRef<'a, D>;

    fn deref(&self) -> &Self::Target {
        // Safety: SnapshotDiffRef and SnapshotDiffRefMut have the same memory layout
        // and casting from mutable to immutable is always safe
        unsafe { mem::transmute(self) }
    }
}