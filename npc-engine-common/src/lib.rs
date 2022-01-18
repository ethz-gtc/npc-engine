mod task;
mod active_task;
mod domain;
mod behavior;
mod config;
mod node;
mod edge;
mod state_diff;
mod util;

pub use domain::*;
pub use task::*;
pub use behavior::*;
pub use config::*;
pub use node::*;
pub use edge::*;
pub use state_diff::*;
pub use util::*;
pub use active_task::*;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct AgentId(pub u32);

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
