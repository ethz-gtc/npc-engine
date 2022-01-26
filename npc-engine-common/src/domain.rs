use std::collections::{BTreeSet, BTreeMap};
use std::hash::Hash;

use ordered_float::NotNan;
use rand_chacha::ChaCha8Rng;

use crate::{AgentId, Behavior, Task, Node, MCTSConfiguration, StateDiffRef};

pub type AgentValue = NotNan<f32>;

/// A domain on which the MCTS planner can plan
pub trait Domain: Sized + 'static {
    /// The state the MCTS plans on.
    type State: std::fmt::Debug + Sized;
    /// A compact set of changes towards a `State` that are accumulated throughout planning.
    type Diff: std::fmt::Debug + Default + Clone + Hash + Eq;
    /// A representation of a display action that can be fetched from a task.
    /// We need Default trait for creating the DisplayAction for the idle placeholder task.
    type DisplayAction: std::fmt::Debug + Default;

    /// Returns all behaviors available for this domain.
    fn list_behaviors() -> &'static [&'static dyn Behavior<Self>];

    /// Gets the current value of the given agent in the given tick and world state.
    fn get_current_value(tick: u64, state_diff: StateDiffRef<Self>, agent: AgentId) -> AgentValue;

    /// Updates the list of agents which are in the horizon of the given agent in the given tick and world state.
    fn update_visible_agents(tick: u64, state_diff: StateDiffRef<Self>, agent: AgentId, agents: &mut BTreeSet<AgentId>);

    /// Gets all agents which are in the horizon of the given agent in the given tick and world state.
    fn get_visible_agents(tick: u64, initial_state: &Self::State, diff: &Self::Diff, agent: AgentId) -> BTreeSet<AgentId> {
        let mut agents = BTreeSet::new();
        let state_diff = StateDiffRef::new(initial_state, diff);
        Self::update_visible_agents(tick, state_diff, agent, &mut agents);
        agents
    }

    /// Gets all possible valid tasks for a given agent in a given tick and world state.
    fn get_tasks(
        tick: u64,
        state_diff: StateDiffRef<'_, Self>,
        agent: AgentId
    ) -> Vec<Box<dyn Task<Self>>> {
        let mut actions = Vec::new();
        Self::list_behaviors()
            .iter()
            .filter(|behavior| behavior.is_valid(tick, state_diff, agent))
            .for_each(|behavior| behavior.add_tasks(tick, state_diff, agent, &mut actions));

        actions.dedup();
        actions
    }

    /// Gets a textual description of the given agent in the given tick and world state.
    /// This will be used by the graph tool to show in each node.
    #[cfg(feature = "graphviz")]
    fn get_state_description(_tick: u64, _state_diff: StateDiffRef<Self>, _agent: AgentId) -> String {
        String::new()
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