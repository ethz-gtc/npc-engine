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

    /// Returns all behaviors available for this domain.
    fn list_behaviors() -> &'static [&'static dyn Behavior<Self>];

    /// Derives a new `Snapshot` for the given agent from the given world state.
    fn derive_snapshot(state: &Self::State, agent: AgentId) -> Self::Snapshot;

    /// Gets the current value of the given agent in the given world state.
    fn get_current_value(state: StateRef<Self>, agent: AgentId) -> f32;

    /// Updates the list of agents which are in the horizon of the given agent in the given world state.
    fn update_visible_agents(state: StateRef<Self>, agent: AgentId, agents: &mut BTreeSet<AgentId>);

    /// Gets all agents which are in the horizon of the given agent in the given world state.
    fn get_visible_agents(state: StateRef<Self>, agent: AgentId) -> BTreeSet<AgentId> {
        let mut agents = BTreeSet::new();
        Self::update_visible_agents(state, agent, &mut agents);
        agents
    }

    /// Gets all possible valid tasks for a given agent in a given world state.
    fn get_tasks<'a>(
        state: StateRef<'a, Self>,
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

pub enum StateRef<'a, D: Domain> {
    State {
        state: &'a D::State,
    },
    Snapshot {
        snapshot: &'a D::Snapshot,
        diff: &'a D::Diff,
    },
}

impl<D: Domain> Copy for StateRef<'_, D> {}
impl<D: Domain> Clone for StateRef<'_, D> {
    fn clone(&self) -> Self {
        match self {
            StateRef::State { state } => StateRef::State { state },
            StateRef::Snapshot { snapshot, diff } => StateRef::Snapshot { snapshot, diff },
        }
    }
}

impl<'a, D: Domain> StateRef<'a, D> {
    pub fn state(state: &'a D::State) -> Self {
        StateRef::State { state }
    }

    pub fn snapshot(snapshot: &'a D::Snapshot, diff: &'a D::Diff) -> Self {
        StateRef::Snapshot { snapshot, diff }
    }
}

impl<D: Domain> fmt::Debug for StateRef<'_, D>
where
    D::State: fmt::Debug,
    D::Snapshot: fmt::Debug,
    D::Diff: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StateRef::State { state } => f.debug_tuple("State").field(state).finish(),
            StateRef::Snapshot { snapshot, diff } => f
                .debug_struct("Snapshot")
                .field("Snapshot", snapshot)
                .field("Diff", diff)
                .finish(),
        }
    }
}

pub enum StateRefMut<'a, D: Domain> {
    State {
        state: &'a mut D::State,
    },
    Snapshot {
        snapshot: &'a D::Snapshot,
        diff: &'a mut D::Diff,
    },
}

impl<'a, D: Domain> StateRefMut<'a, D> {
    pub fn state(state: &'a mut D::State) -> Self {
        StateRefMut::State { state }
    }

    pub fn snapshot(snapshot: &'a D::Snapshot, diff: &'a mut D::Diff) -> Self {
        StateRefMut::Snapshot { snapshot, diff }
    }
}

impl<'a, D: Domain> Deref for StateRefMut<'a, D> {
    type Target = StateRef<'a, D>;

    fn deref(&self) -> &Self::Target {
        // Safety: StateRef and StateRefMut have the same memory layout
        // and casting from mutable to immutable is always safe
        unsafe { mem::transmute(self) }
    }
}