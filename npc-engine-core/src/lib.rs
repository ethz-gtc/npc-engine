/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

//! This is the core of the NPC engine, containing the [MCTS] algorithm implementation and related abstractions.
//!
//! We provide several [examples](https://github.com/ethz-gtc/npc-engine/tree/main/npc-engine-core/examples)
//! as introductions on how to use the planner.
//! A good place to start is [tic-tac-toe](https://github.com/ethz-gtc/npc-engine/tree/main/npc-engine-core/examples/tic-tac-toe).
//!
//! The core of the planner is the [MCTS] struct, which holds the state of the planner.
//! It has two constructors, a simplified one, [new](MCTS::new), and a complete one, [new_with_tasks](MCTS::new_with_tasks).
//! Once constructed, the [run](MCTS::run) method performs the search and returns the best task.
//! After a search, the resulting tree can be inspected, starting from the [root node](MCTS::root_node).
//!
//! The [MCTS] struct is generic over a [Domain], which you have to implement to describe your own planning domain.
//! You need to implement at least these three methods:
//! * [list_behaviors](Domain::list_behaviors) returns the possible actions employing a hierarchical [Behavior] abstraction.
//! * [get_current_value](Domain::get_current_value) returns the instantaneous (not discounted) value of an agent in a given state.
//! * [update_visible_agents](Domain::update_visible_agents) lists all agents visible from a given agent in a given state.
//!
//! The `graphviz` feature enables to output the search tree in the Graphviz's dot format using the [plot_mcts_tree](graphviz::plot_mcts_tree) function.
//!
//! Additional features and utilites such as execution loops are available in the [`npc-engine-utils`](https://crates.io/crates/npc-engine-utils/) crate.
//! You might want to use them in your project as they make the planner significantly simpler to use.
//! Most [examples](https://github.com/ethz-gtc/npc-engine/tree/main/npc-engine-core/examples) use them.

mod active_task;
mod behavior;
mod config;
mod domain;
mod edge;
mod mcts;
mod node;
mod state_diff;
mod task;
mod util;

pub use active_task::*;
pub use behavior::*;
pub use config::*;
pub use domain::*;
pub use edge::*;
pub use mcts::*;
pub use node::*;
pub use state_diff::*;
pub use task::*;
use util::*;

/// The identifier of an agent, essentially a u32.
#[derive(
    Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "kebab-case")]
pub struct AgentId(
    /// The internal identifier
    pub u32,
);
impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "A{}", self.0)
    }
}
