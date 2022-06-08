/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::collections::{BTreeMap, BTreeSet};
use std::hash::Hash;

use ordered_float::NotNan;
use rand_chacha::ChaCha8Rng;

use crate::{AgentId, Behavior, Edges, MCTSConfiguration, Node, StateDiffRef, Task};

/// The "current" value an agent has in a given state.
pub type AgentValue = NotNan<f32>;

/// A domain on which the MCTS planner can plan.
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
    fn update_visible_agents(
        start_tick: u64,
        tick: u64,
        state_diff: StateDiffRef<Self>,
        agent: AgentId,
        agents: &mut BTreeSet<AgentId>,
    );

    /// Gets all possible valid tasks for a given agent in a given tick and world state.
    fn get_tasks(
        tick: u64,
        state_diff: StateDiffRef<'_, Self>,
        agent: AgentId,
    ) -> Vec<Box<dyn Task<Self>>> {
        let mut actions = Vec::new();
        Self::list_behaviors()
            .iter()
            .for_each(|behavior| behavior.add_tasks(tick, state_diff, agent, &mut actions));

        actions.dedup();
        actions
    }

    /// Gets a textual description of the given world state.
    /// This will be used by the graph tool to show in each node, and the log tool to dump the state.
    fn get_state_description(_state_diff: StateDiffRef<Self>) -> String {
        String::new()
    }

    /// Gets the new agents present in a diff but not in a state.
    fn get_new_agents(_state_diff: StateDiffRef<Self>) -> Vec<AgentId> {
        vec![]
    }

    /// Gets the display actions for idle task.
    fn display_action_task_idle() -> Self::DisplayAction {
        Default::default()
    }

    /// Gets the display actions for planning task.
    fn display_action_task_planning() -> Self::DisplayAction {
        Default::default()
    }
}

/// An estimator of state-value function.
pub trait StateValueEstimator<D: Domain>: Send {
    /// Takes the state of an explored node and returns the estimated expected (discounted) values.
    ///
    /// Returns None if the passed node has no unexpanded edge.
    #[allow(clippy::too_many_arguments)]
    fn estimate(
        &mut self,
        rnd: &mut ChaCha8Rng,
        config: &MCTSConfiguration,
        initial_state: &D::State,
        start_tick: u64,
        node: &Node<D>,
        edges: &Edges<D>,
        depth: u32,
    ) -> Option<BTreeMap<AgentId, f32>>;
}
