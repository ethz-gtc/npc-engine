use std::{hash::{Hash, Hasher}};

use downcast_rs::{impl_downcast, Downcast};

use crate::{AgentId, Domain, StateDiffRef, StateDiffRefMut, impl_task_boxed_methods};

pub type TaskDuration = u64;

/// A task that modifies the state.
/// It is illegal to have a task of both 0-duration and not modifying the state,
/// as this would lead to self-looping nodes in the planner.
pub trait Task<D: Domain>: std::fmt::Debug + Downcast + Send + Sync {
    /// Returns the relative weight of the task for the given agent in the given tick and world state, by default weight is 1.0
    fn weight(&self, _tick: u64, _state_diff: StateDiffRef<D>, _agent: AgentId) -> f32 {
        1.0
    }

    /// Returns the duration of the task, for a given agent in a given tick and world state
    fn duration(&self, _tick: u64, _state_diff: StateDiffRef<D>, _agent: AgentId) -> TaskDuration;

    /// Executes one step of the task for the given agent on the given tick and world state.
    fn execute(&self, tick: u64, state_diff: StateDiffRefMut<D>, agent: AgentId) -> Option<Box<dyn Task<D>>>;

    /// Returns if the task is valid for the given agent in the given tick and world state.
    fn is_valid(&self, tick: u64, state_diff: StateDiffRef<D>, agent: AgentId) -> bool;

    /// Returns the display actions corresponding to this task.
    fn display_action(&self) -> D::DisplayAction;

    /// Utility method for cloning, since `Self: Clone` is not object-safe
    fn box_clone(&self) -> Box<dyn Task<D>>;

    /// Utility method for hashing, since `Self: Hash` is not object-safe
    fn box_hash(&self, state: &mut dyn Hasher);

    /// Utility method for equality, since trait objects are not inherently `Eq`.
    /// Should perform downcast to current type and then check equality.
    #[allow(clippy::borrowed_box)]
    fn box_eq(&self, other: &Box<dyn Task<D>>) -> bool;
}

/// An idle task of duration 1 that is used by the planner when the task of an agent is not known
#[derive(Debug, Hash, Clone, PartialEq)]
pub struct IdleTask;

impl<D: Domain> Task<D> for IdleTask {
    fn weight(&self, _tick: u64, _state_diff: StateDiffRef<D>, _agent: AgentId) -> f32 {
        1f32
    }

    fn duration(&self, _tick: u64, _state_diff: StateDiffRef<D>, _agent: AgentId) -> TaskDuration {
        1
    }

    fn execute(&self, _tick: u64, _state_diff: StateDiffRefMut<D>, _agent: AgentId) -> Option<Box<dyn Task<D>>> {
        None
    }

    fn is_valid(&self, _tick: u64, _state_diff: StateDiffRef<D>, _agent: AgentId) -> bool {
        true
    }

    fn display_action(&self) -> D::DisplayAction {
        Default::default()
    }

    impl_task_boxed_methods!(D);
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
    
        fn box_hash(&self, mut state: &mut dyn std::hash::Hasher) {
            use std::hash::Hash;
            self.hash(&mut state)
        }
    
        fn box_eq(&self, other: &Box<dyn Task<$e>>) -> bool {
            other.downcast_ref::<Self>().map_or(false, |other| self.eq(other))
        }
    };
}