//! This is the utility module of the NPC engine, containing re-usable support code.
//!
//! It contains the following features:
//! - An helper trait when the diff is just a copy of the domain.
//! - Two "executors" that implement the execution logic of a domain beyond planning itself, and related abstractions.
//! - A simple implementation of feed-forward leaky ReLU neurons including learning based on back-propagation.
//! - Simple 2-D coordinates and direction implementations.
//! - Some helper functions to plot trees.
//! - Some helper functions to simplify functional programming.

mod functional;
mod coord2d;
mod direction;
mod global_domain;
mod option_state_diff;
mod graphs;
mod executor;
mod neuron;

pub use functional::*;
pub use coord2d::*;
pub use direction::*;
pub use global_domain::*;
pub use option_state_diff::*;
pub use graphs::*;
pub use executor::*;
pub use neuron::*;