/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::{
    hash::{Hash, Hasher},
    num::NonZeroU64,
};

use downcast_rs::{impl_downcast, Downcast};

use crate::{impl_task_boxed_methods, Context, ContextMut, Domain};

/// The duration of a task, in ticks.
pub type TaskDuration = u64;

/// Transforms the debug string of a task to a string that can safely be used for filenames.
pub fn debug_name_to_filename_safe(debug_name: &str) -> String {
    debug_name
        .replace([' ', '(', ')'], "")
        .replace('{', "_")
        .replace('}', "")
        .replace([' ', ':', ','], "_")
}

/// A task that modifies the state.
///
/// It is illegal to have a task of both 0-duration and not modifying the state,
/// as this would lead to self-looping nodes in the planner.
pub trait Task<D: Domain>: std::fmt::Debug + Downcast + Send + Sync {
    /// Returns the relative weight of the task for the given agent in the given tick and world state, by default weight is 1.0.
    fn weight(&self, _ctx: Context<D>) -> f32 {
        1.0
    }

    /// Returns the duration of the task, for a given agent in a given tick and world state.
    fn duration(&self, ctx: Context<D>) -> TaskDuration;

    /// Executes one step of the task for the given agent on the given tick and world state.
    fn execute(&self, ctx: ContextMut<D>) -> Option<Box<dyn Task<D>>>;

    /// Returns if the task is valid for the given agent in the given tick and world state.
    fn is_valid(&self, ctx: Context<D>) -> bool;

    /// Returns the display actions corresponding to this task.
    fn display_action(&self) -> D::DisplayAction;

    /// Utility method for cloning, since `Self: Clone` is not object-safe.
    ///
    /// Use the macro [impl_task_boxed_methods] to automatically generate this method.
    fn box_clone(&self) -> Box<dyn Task<D>>;

    /// Utility method for hashing, since `Self: Hash` is not object-safe.
    ///
    /// Use the macro [impl_task_boxed_methods] to automatically generate this method.
    fn box_hash(&self, state: &mut dyn Hasher);

    /// Utility method for equality, since trait objects are not inherently `Eq`.
    ///
    /// Should perform downcast to current type and then check equality.
    ///
    /// Use the macro [impl_task_boxed_methods] to automatically generate this method.
    #[allow(clippy::borrowed_box)]
    fn box_eq(&self, other: &Box<dyn Task<D>>) -> bool;
}

/// An idle task of duration 1 that is used by the planner when the task of an agent is not known.
#[derive(Debug, Hash, Clone, PartialEq, Eq)]
pub struct IdleTask;

impl<D: Domain> Task<D> for IdleTask {
    fn weight(&self, _ctx: Context<D>) -> f32 {
        1f32
    }

    fn duration(&self, _ctx: Context<D>) -> TaskDuration {
        1
    }

    fn execute(&self, _ctx: ContextMut<D>) -> Option<Box<dyn Task<D>>> {
        None
    }

    fn is_valid(&self, _ctx: Context<D>) -> bool {
        true
    }

    fn display_action(&self) -> D::DisplayAction {
        D::display_action_task_idle()
    }

    impl_task_boxed_methods!(D);
}

/// A task to represent planning in the planning tree, if these need to be represented.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct PlanningTask(
    /// The duration of the planning task
    pub NonZeroU64,
);

impl<D: Domain> Task<D> for PlanningTask {
    fn weight(&self, _ctx: Context<D>) -> f32 {
        1f32
    }

    fn duration(&self, _ctx: Context<D>) -> TaskDuration {
        self.0.get()
    }

    fn execute(&self, _ctx: ContextMut<D>) -> Option<Box<dyn Task<D>>> {
        None
    }

    fn is_valid(&self, _ctx: Context<D>) -> bool {
        true
    }

    fn display_action(&self) -> D::DisplayAction {
        D::display_action_task_planning()
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

/// Task implementors can use this macro to implement the `box_clone`, `box_hash` and `box_eq` functions.
///
/// The parameter is the name of your [Domain] struct.
#[macro_export]
macro_rules! impl_task_boxed_methods {
    ($domain: ty) => {
        fn box_clone(&self) -> Box<dyn Task<$domain>> {
            Box::new(self.clone())
        }

        fn box_hash(&self, mut state: &mut dyn std::hash::Hasher) {
            use std::hash::Hash;
            self.hash(&mut state)
        }

        fn box_eq(&self, other: &Box<dyn Task<$domain>>) -> bool {
            other
                .downcast_ref::<Self>()
                .map_or(false, |other| self.eq(other))
        }
    };
}
