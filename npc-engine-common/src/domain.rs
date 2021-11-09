use std::collections::BTreeSet;
use std::hash::Hash;
use std::ops::Deref;
use std::{fmt, mem};

use crate::{AgentId, Behavior, Task};

// TODO: remove debug constraints
pub trait NpcEngine: Sized + 'static {
    type State: std::fmt::Debug + Sized + 'static;
    type Snapshot: std::fmt::Debug + Sized + 'static;
    type Diff: std::fmt::Debug + Default + Clone + Hash + Eq + 'static;

    fn behaviors() -> &'static [&'static dyn Behavior<Self>];

    fn derive(state: &Self::State, agent: AgentId) -> Self::Snapshot;

    fn apply(snapshot: &mut Self::Snapshot, diff: &Self::Diff);

    fn compatible(snapshot: &Self::Snapshot, other: &Self::Snapshot, agent: AgentId) -> bool;

    fn value(state: StateRef<Self>, agent: AgentId) -> f32;

    fn agents(state: StateRef<Self>, agent: AgentId, agents: &mut BTreeSet<AgentId>);

    fn add_tasks<'a>(
        state: StateRef<'a, Self>,
        agent: AgentId,
        actions: &mut Vec<Box<dyn Task<Self>>>,
    ) {
        Self::behaviors()
            .iter()
            .filter(|behavior| behavior.predicate(state, agent))
            .for_each(|behavior| behavior.add_tasks(state, agent, actions));

        actions.dedup();
    }
}

pub enum StateRef<'a, E: NpcEngine> {
    State {
        state: &'a E::State,
    },
    Snapshot {
        snapshot: &'a E::Snapshot,
        diff: &'a E::Diff,
    },
}

impl<E: NpcEngine> Copy for StateRef<'_, E> {}
impl<E: NpcEngine> Clone for StateRef<'_, E> {
    fn clone(&self) -> Self {
        match self {
            StateRef::State { state } => StateRef::State { state },
            StateRef::Snapshot { snapshot, diff } => StateRef::Snapshot { snapshot, diff },
        }
    }
}

impl<'a, E: NpcEngine> StateRef<'a, E> {
    pub fn state(state: &'a E::State) -> Self {
        StateRef::State { state }
    }

    pub fn snapshot(snapshot: &'a E::Snapshot, diff: &'a E::Diff) -> Self {
        StateRef::Snapshot { snapshot, diff }
    }
}

impl<E: NpcEngine> fmt::Debug for StateRef<'_, E>
where
    E::State: fmt::Debug,
    E::Snapshot: fmt::Debug,
    E::Diff: fmt::Debug,
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

pub enum StateRefMut<'a, E: NpcEngine> {
    State {
        state: &'a mut E::State,
    },
    Snapshot {
        snapshot: &'a E::Snapshot,
        diff: &'a mut E::Diff,
    },
}

impl<'a, E: NpcEngine> StateRefMut<'a, E> {
    pub fn state(state: &'a mut E::State) -> Self {
        StateRefMut::State { state }
    }

    pub fn snapshot(snapshot: &'a E::Snapshot, diff: &'a mut E::Diff) -> Self {
        StateRefMut::Snapshot { snapshot, diff }
    }
}

impl<'a, E: NpcEngine> Deref for StateRefMut<'a, E> {
    type Target = StateRef<'a, E>;

    fn deref(&self) -> &Self::Target {
        // Safety: StateRef and StateRefMut have the same memory layout
        // and casting from mutable to immutable is always safe
        unsafe { mem::transmute(self) }
    }
}