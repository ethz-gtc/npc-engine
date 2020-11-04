use downcast_rs::{impl_downcast, Downcast};
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::{AgentId, NpcEngine, StateRef, StateRefMut};

pub trait Task<E: NpcEngine>: fmt::Display + Downcast + Send + Sync {
    /// Returns the relative weight of the task.
    fn weight(&self, state: StateRef<E>, agent: AgentId) -> f32;

    /// Executes one step of the task on the state.
    fn execute(&self, state: StateRefMut<E>, agent: AgentId) -> Option<Box<dyn Task<E>>>;

    /// Returns `true` if the task is still valid for the state.
    fn valid(&self, state: StateRef<E>, agent: AgentId) -> bool;

    /// Utility method for cloning, since `Self: Clone` is not object-safe
    fn box_clone(&self) -> Box<dyn Task<E>>;

    /// Utility method for hashing, since `Self: Hash` is not object-safe
    fn box_hash(&self, state: &mut dyn Hasher);

    /// Utility method for equality, since trait objects are not inherently `Eq`.
    /// Should perform downcast to current type and then check equality.
    fn box_eq(&self, other: &Box<dyn Task<E>>) -> bool;
}

impl_downcast!(Task<E> where E: NpcEngine);

impl<E: NpcEngine> Clone for Box<dyn Task<E>> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

impl<E: NpcEngine> Hash for Box<dyn Task<E>> {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.box_hash(state);
    }
}

impl<E: NpcEngine> PartialEq for Box<dyn Task<E>> {
    fn eq(&self, other: &Self) -> bool {
        self.box_eq(other)
    }
}

impl<E: NpcEngine> Eq for Box<dyn Task<E>> {}
