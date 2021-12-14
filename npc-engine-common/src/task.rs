use std::fmt;
use std::hash::{Hash, Hasher};

use downcast_rs::{impl_downcast, Downcast};

use crate::{AgentId, Domain, StateDiffRef, StateDiffRefMut};

pub trait Task<D: Domain>: fmt::Display + Downcast + Send + Sync {
    /// Returns the relative weight of the task for the given agent in the given world state.
    fn weight(&self, state: StateDiffRef<D>, agent: AgentId) -> f32;

    /// Executes one step of the task for the given agent on the given world state.
    fn execute(&self, state: StateDiffRefMut<D>, agent: AgentId) -> Option<Box<dyn Task<D>>>;

    /// Returns if the task is valid for the given agent in the given world state.
    fn is_valid(&self, state: StateDiffRef<D>, agent: AgentId) -> bool;

    /// Returns the display actions corresponding to this task.
    fn display_action(&self) -> D::DisplayAction;

    /// Utility method for cloning, since `Self: Clone` is not object-safe
    fn box_clone(&self) -> Box<dyn Task<D>>;

    /// Utility method for hashing, since `Self: Hash` is not object-safe
    fn box_hash(&self, state: &mut dyn Hasher);

    /// Utility method for equality, since trait objects are not inherently `Eq`.
    /// Should perform downcast to current type and then check equality.
    fn box_eq(&self, other: &Box<dyn Task<D>>) -> bool;
}

impl_downcast!(Task<D> where D: Domain);

impl<D: Domain> Clone for Box<dyn Task<D>> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

impl<D: Domain> Hash for Box<dyn Task<D>> {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.box_hash(state);
    }
}

impl<D: Domain> PartialEq for Box<dyn Task<D>> {
    fn eq(&self, other: &Self) -> bool {
        self.box_eq(other)
    }
}

impl<D: Domain> Eq for Box<dyn Task<D>> {}

#[macro_export]
macro_rules! impl_task_boxed_methods {
    ($e: ty) => {
        fn box_clone(&self) -> Box<dyn Task<$e>> {
            Box::new(self.clone())
        }
    
        fn box_hash(&self, mut state: &mut dyn Hasher) {
            self.hash(&mut state)
        }
    
        fn box_eq(&self, other: &Box<dyn Task<$e>>) -> bool {
            other.downcast_ref::<Self>().map_or(false, |other| self.eq(other))
        }
    };
}