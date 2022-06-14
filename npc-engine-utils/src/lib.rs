/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

//! This is the utility module of the NPC engine, containing re-usable support code.
//!
//! It contains the following features:
//! - An helper trait ([OptionDiffDomain]) when the diff is just a copy of the domain.
//! - Two "executors" ([SimpleExecutor] and [ThreadedExecutor]) that implement the execution logic of a domain beyond planning itself, and related abstractions.
//! - A simple implementation of feed-forward leaky ReLU neurons ([Neuron]) and simple networks of these ([NeuralNetwork]), including learning based on back-propagation ([NeuralNetwork::train]).
//! - Simple 2-D coordinates ([Coord2D]) and direction ([Direction]) implementations.
//! - Some helper functions to plot trees ([plot_tree_in_tmp] and [plot_tree_in_tmp_with_task_name]).
//! - Some helper functions to simplify functional programming with tuples (such as [keep_first], etc.).

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
