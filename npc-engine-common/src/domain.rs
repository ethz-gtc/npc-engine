use std::collections::{BTreeSet, BTreeMap};
use std::hash::Hash;

use rand_chacha::ChaCha8Rng;

use crate::{AgentId, Behavior, Task, Node, MCTSConfiguration, StateDiffRef};

// TODO: remove debug constraints
/// A domain on which the MCTS planner can plan
pub trait Domain: Sized + 'static {
    /// The state the MCTS plans on.
    type State: std::fmt::Debug + Sized + 'static;
    /// A compact set of changes towards a `State` that are accumulated throughout planning.
    type Diff: std::fmt::Debug + Default + Clone + Hash + Eq + 'static;
    /// A representation of a display action that can be fetched from a task.
    type DisplayAction;

    /// Returns all behaviors available for this domain.
    fn list_behaviors() -> &'static [&'static dyn Behavior<Self>];

    /// Gets the current value of the given agent in the given world state.
    fn get_current_value(state_diff: StateDiffRef<Self>, agent: AgentId) -> f32;

    /// Updates the list of agents which are in the horizon of the given agent in the given world state.
    fn update_visible_agents(state_diff: StateDiffRef<Self>, agent: AgentId, agents: &mut BTreeSet<AgentId>);

    /// Gets all agents which are in the horizon of the given agent in the given world state.
    fn get_visible_agents(state_diff: StateDiffRef<Self>, agent: AgentId) -> BTreeSet<AgentId> {
        let mut agents = BTreeSet::new();
        Self::update_visible_agents(state_diff, agent, &mut agents);
        agents
    }

    /// Gets all possible valid tasks for a given agent in a given world state.
    fn get_tasks<'a>(
        state_diff: StateDiffRef<'a, Self>,
        agent: AgentId
    ) -> Vec<Box<dyn Task<Self>>> {
        let mut actions = Vec::new();
        Self::list_behaviors()
            .iter()
            .filter(|behavior| behavior.is_valid(state_diff, agent))
            .for_each(|behavior| behavior.add_tasks(state_diff, agent, &mut actions));

        actions.dedup();
        actions
    }
}

/// Estimator of state-value function: takes state of explored node and returns the estimated expected (discounted) values
pub trait StateValueEstimator<D: Domain> {
    fn estimate(
        &mut self,
        rnd: &mut ChaCha8Rng,
        config: &MCTSConfiguration,
        initial_state: &D::State,
        node: &Node<D>,
        depth: u32,
        agent: AgentId, // agent of the MCTS
    ) -> BTreeMap<AgentId, f32>;
}