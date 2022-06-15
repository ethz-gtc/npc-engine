/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

//! This is the utility module of the [NPC engine](https://crates.io/crates/npc-engine-core/), containing helpful utility code.
//!
//! It contains the following features:
//! - A helper trait [OptionDiffDomain] that can be used when [Diffs](Domain::Diff) are just copies of the [State](Domain::State).
//! - Two executors (update loops), [SimpleExecutor] and [ThreadedExecutor], that implement the execution logic of a [Domain] beyond planning itself, and related abstractions.
//! - A simple implementation of feed-forward leaky ReLU neurons ([Neuron]) and corresponding simple networks ([NeuralNetwork]), providing learning based on back-propagation ([NeuralNetwork::train]).
//! - Simple 2-D coordinates ([Coord2D]) and direction ([Direction]) implementations.
//! - Helper functions to plot search trees: [plot_tree_in_tmp] and [plot_tree_in_tmp_with_task_name].
//! - Helper functions to simplify functional programming with tuples: [keep_first] and [keep_second], and their mutable versions [keep_first_mut] and [keep_second_mut].

#[cfg(doc)]
use npc_engine_core::Domain;

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
