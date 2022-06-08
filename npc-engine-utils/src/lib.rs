/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

//! This is the utility module of the NPC engine, containing re-usable support code.
//!
//! It contains the following features:
//! - An helper trait when the diff is just a copy of the domain.
//! - Two "executors" that implement the execution logic of a domain beyond planning itself, and related abstractions.
//! - A simple implementation of feed-forward leaky ReLU neurons including learning based on back-propagation.
//! - Simple 2-D coordinates and direction implementations.
//! - Some helper functions to plot trees.
//! - Some helper functions to simplify functional programming.

mod coord2d;
mod direction;
mod executor;
mod functional;
mod global_domain;
mod graphs;
mod neuron;
mod option_state_diff;

pub use coord2d::*;
pub use direction::*;
pub use executor::*;
pub use functional::*;
pub use global_domain::*;
pub use graphs::*;
pub use neuron::*;
pub use option_state_diff::*;
