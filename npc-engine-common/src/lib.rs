mod task;
mod active_task;
mod domain;
mod behavior;
mod config;
mod node;
mod edge;
mod state_diff;
mod util;
mod mcts;

pub use domain::*;
pub use task::*;
pub use behavior::*;
pub use config::*;
pub use node::*;
pub use edge::*;
pub use state_diff::*;
pub use util::*;
pub use active_task::*;
pub use mcts::*;

/// The identifier of an agent, essentially a u32.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct AgentId(pub u32);
impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "A{}", self.0)
    }
}
