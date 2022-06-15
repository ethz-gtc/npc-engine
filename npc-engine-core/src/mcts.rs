/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::collections::HashMap;
use std::collections::{btree_map::Entry, BTreeMap, BTreeSet, HashSet};
use std::f32;
use std::mem;
use std::ops::Range;
use std::time::{Duration, Instant};

use rand::{
    distributions::WeightedIndex,
    prelude::{thread_rng, Distribution, RngCore, SeedableRng},
    Rng,
};
use rand_chacha::ChaCha8Rng;

use crate::*;

// TODO: Consider replacing Seeded hashmaps with btreemaps

// TODO: Once is_nan() and unwrap() are const, remove unsafe

// SAFETY: 0 is not NaN
const VALUE_ZERO: AgentValue = unsafe { AgentValue::new_unchecked(0.) };
// SAFETY: INFINITY is not NaN
const VALUE_INFINITE: AgentValue = unsafe { AgentValue::new_unchecked(std::f32::INFINITY) };
// SAFETY: NEG_INFINITY is not NaN
const VALUE_NEG_INFINITE: AgentValue = unsafe { AgentValue::new_unchecked(std::f32::NEG_INFINITY) };

/// The state of a running planner instance.
pub struct MCTS<D: Domain> {
    // Statistics
    time: Duration,

    // Config
    config: MCTSConfiguration,
    state_value_estimator: Box<dyn StateValueEstimator<D> + Send>,
    early_stop_condition: Option<Box<EarlyStopCondition>>,

    // Run-specific parameters
    root_agent: AgentId,
    seed: u64,

    // Nodes
    root: Node<D>,
    nodes: SeededHashMap<Node<D>, Edges<D>>,

    // Globals
    q_value_ranges: BTreeMap<AgentId, Range<AgentValue>>,

    // State before planning
    initial_state: D::State,
    start_tick: u64,

    // Rng
    rng: ChaCha8Rng,
}

/// The possible outcomes from a tree policy pass.
enum TreePolicyOutcome<D: Domain> {
    NodeCreated(u32, Node<D>, Vec<Edge<D>>), // depth, new node, path
    NoValidTask(u32, Vec<Edge<D>>),          // depth, path
    NoChildNode(u32, Node<D>, Vec<Edge<D>>), // depth, node, path
    DepthLimitReached(u32, Node<D>, Vec<Edge<D>>), // depth, new node, path
}

impl<D: Domain> MCTS<D> {
    /// Instantiates a new search tree for the given state, with idle tasks for all agents and starting at tick 0.
    pub fn new(initial_state: D::State, root_agent: AgentId, config: MCTSConfiguration) -> Self {
        let state_value_estimator = Box::new(DefaultPolicyEstimator {});
        Self::new_with_tasks(
            initial_state,
            root_agent,
            0,
            Default::default(),
            config,
            state_value_estimator,
            None,
        )
    }

    /// Instantiates a new search tree for the given state, with active tasks for all agents and starting at a given tick.
    pub fn new_with_tasks(
        initial_state: D::State,
        root_agent: AgentId,
        start_tick: u64,
        tasks: ActiveTasks<D>,
        config: MCTSConfiguration,
        state_value_estimator: Box<dyn StateValueEstimator<D> + Send>,
        early_stop_condition: Option<Box<EarlyStopCondition>>,
    ) -> Self {
        // Check whether there is a task for this agent already
        let next_task =
            get_task_for_agent(&tasks, root_agent).map(|active_task| active_task.task.clone());

        // Create new root node
        let root = Node::new(NodeInner::new(
            &initial_state,
            start_tick,
            Default::default(),
            root_agent,
            start_tick,
            tasks,
        ));

        // Prepare nodes, reserve the maximum amount we could need
        let mut nodes = SeededHashMap::with_capacity_and_hasher(
            config.visits as usize + 1,
            SeededRandomState::default(),
        );

        // Insert new root node
        let root_edges = Edges::new(&root, &initial_state, next_task);
        nodes.insert(root.clone(), root_edges);

        // Compute seed
        let cur_seed = config.seed.unwrap_or_else(|| thread_rng().next_u64());

        MCTS {
            time: Duration::default(),
            config,
            state_value_estimator,
            early_stop_condition,
            seed: cur_seed,
            root_agent,
            root,
            nodes,
            q_value_ranges: Default::default(),
            initial_state,
            start_tick,
            rng: ChaCha8Rng::seed_from_u64(cur_seed),
        }
    }

    /// Returns the best task, using exploration value of 0.
    pub fn best_task_at_root(&mut self) -> Option<Box<dyn Task<D>>> {
        let range = self.min_max_range(self.root_agent);
        let edges = self.nodes.get(&self.root).unwrap();
        edges
            // Get best expanded tasks.
            .best_task(self.root_agent, 0., range)
            // If none, sample unexpanded tasks.
            .or_else(|| {
                edges.unexpanded_tasks.as_ref().and_then(|(_, tasks)| {
                    if tasks.is_empty() {
                        None
                    } else {
                        let index = self.rng.gen_range(0..tasks.len());
                        Some(tasks[index].clone())
                    }
                })
            })
    }

    /// Returns the best task, following a given recent task history, in case planning tasks are used.
    pub fn best_task_with_history(
        &self,
        task_history: &HashMap<AgentId, ActiveTask<D>>,
    ) -> Box<dyn Task<D>>
    where
        D: DomainWithPlanningTask,
    {
        log::debug!(
            "Finding best task for {} using history {:?}",
            self.root_agent,
            task_history
        );
        let mut current_node = self.root.clone();
        let mut edges = self.nodes.get(&current_node).unwrap();
        let mut depth = 0;
        loop {
            let node_agent = current_node.agent();
            let node_tick = current_node.tick();
            let edge = if edges.expanded_tasks.len() == 1 {
                let (task, edge) = edges.expanded_tasks.iter().next().unwrap();
                log::trace!("[{depth}] T{node_tick} {node_agent} skipping {task:?}");

                // Skip non-branching nodes
                edge
            } else {
                let executed_task = task_history.get(&node_agent);
                let executed_task = executed_task
                    .unwrap_or_else(|| panic!("Found no task for {node_agent} is history"));
                let task = &executed_task.task;
                log::trace!("[{depth}] T{node_tick} {node_agent} executed {task:?}");

                let edge = edges.expanded_tasks.get(task);

                if edge.is_none() {
                    log::info!("{node_agent} executed unexpected {task:?} not present in search tree, returning fallback task");
                    return D::fallback_task(self.root_agent);
                }

                edge.unwrap()
            };
            let edge = edge.lock().unwrap();
            current_node = edge.child();
            // log::debug!("NEW_CUR_NODE: {current_node:?} {:p}", Arc::as_ptr(current_node));
            edges = self.nodes.get(&current_node).unwrap();

            depth += 1;

            // Stop if we reach our own node again
            if current_node.agent() == self.root_agent {
                break;
            }
        }

        // Return best task, using exploration value of 0
        let range = self.min_max_range(self.root_agent);
        let best = edges.best_task(self.root_agent, 0., range);

        if best.is_none() {
            log::info!(
                "No valid task for agent {}, returning fallback task",
                self.root_agent
            );
            return D::fallback_task(self.root_agent);
        }

        best.unwrap().clone()
    }

    /// Returns the q-value at root.
    pub fn q_value_at_root(&self, agent: AgentId) -> f32 {
        let edges = self.nodes.get(&self.root).unwrap();
        edges.q_value((0, 0.), agent).unwrap()
    }

    /// Executes the MCTS search.
    ///
    /// Returns the current best task, if there is at least one task for the root node.
    pub fn run(&mut self) -> Option<Box<dyn Task<D>>> {
        // Reset globals
        self.q_value_ranges.clear();

        let start = Instant::now();
        let max_visits = self.config.visits;
        for i in 0..max_visits {
            // Execute tree policy, if expansion resulted in no node, do nothing
            let tree_policy_outcome = self.tree_policy();

            // Only if the tree policy resulted in a node expansion, we execute the default policy,
            // but in any case we update the visit count.
            let (path, rollout_values) = match tree_policy_outcome {
                TreePolicyOutcome::NodeCreated(depth, leaf, path) => {
                    // Execute default policy
                    let edges = self.nodes.get(&leaf).unwrap();
                    let rollout_values = self.state_value_estimator.estimate(
                        &mut self.rng,
                        &self.config,
                        &self.initial_state,
                        self.start_tick,
                        &leaf,
                        edges,
                        depth,
                    );
                    (path, rollout_values)
                }
                TreePolicyOutcome::NoValidTask(_, path) => (path, None),
                TreePolicyOutcome::NoChildNode(_, _, path) => (path, None),
                TreePolicyOutcome::DepthLimitReached(_, _, path) => (path, None),
            };

            // Backpropagate results
            self.backpropagation(path, rollout_values);

            // Early stopping if told so by some user-defined condition
            if let Some(early_stop_condition) = &self.early_stop_condition {
                if early_stop_condition() {
                    log::info!("{:?} early stops planning after {} visits", self.agent(), i);
                    break;
                }
            }
        }
        self.time = start.elapsed();

        self.best_task_at_root()
    }

    /// MCTS tree policy. Executes the `selection` and `expansion` phases.
    fn tree_policy(&mut self) -> TreePolicyOutcome<D> {
        let agents = self.root.agents();

        let mut node = self.root.clone();

        // Path through the tree, including root and leaf
        let mut path = Vec::with_capacity(self.config.depth as usize * agents.len());

        // Execute selection until at most `depth`, expressed as number of ticks
        let mut depth = 0;
        while depth < self.config.depth {
            let mut edges = self.nodes.get_mut(&node).unwrap();

            // -------------------------
            // Expansion
            // -------------------------
            // If weights are non-empty, the node has not been fully expanded
            if let Some((weights, tasks)) = edges.unexpanded_tasks.as_mut() {
                // Clone a new diff from the current one to be used for the newly expanded node
                let mut diff = node.diff.clone();

                // Select expansion task randomly
                let idx = weights.sample(&mut self.rng);
                let task = tasks[idx].clone();
                let state_diff = StateDiffRef::new(&self.initial_state, &diff);
                debug_assert!(task.is_valid(node.tick, state_diff, node.active_agent));
                log::debug!(
                    "T{}\t{:?} - Expand task: {:?}",
                    node.tick,
                    node.active_agent,
                    task
                );

                // Set weight of chosen task to zero to mark it as expanded.
                // As updating weights returns an error if all weights are zero,
                // we have to handle this by setting unexpanded_tasks to None if we get an error.
                if weights.update_weights(&[(idx, &0.)]).is_err() {
                    // All weights being zero implies the node is fully expanded
                    edges.unexpanded_tasks = None;
                }

                // Clone active tasks for child node, removing task of active agent
                let mut child_tasks = node
                    .tasks
                    .iter()
                    .filter(|task| task.agent != node.active_agent)
                    .cloned()
                    .collect::<BTreeSet<_>>();
                // Create and insert new active task for the active agent and the selected task
                let active_task =
                    ActiveTask::new(node.active_agent, task.clone(), node.tick, state_diff);
                child_tasks.insert(active_task);
                log::trace!("\tActive Tasks ({}):", child_tasks.len());
                for active_task in &child_tasks {
                    log::trace!(
                        "\t  {:?}: {:?} ends T{}",
                        active_task.agent,
                        active_task.task,
                        active_task.end
                    );
                }

                // Get task that finishes in the next node
                let next_active_task = child_tasks.iter().next().unwrap().clone();
                log::trace!(
                    "\tNext Active Task: {:?}: {:?} ends T{}",
                    next_active_task.agent,
                    next_active_task.task,
                    next_active_task.end
                );

                // If it is not valid, abort this expansion
                let is_task_valid = next_active_task.task.is_valid(
                    next_active_task.end,
                    state_diff,
                    next_active_task.agent,
                );
                if !is_task_valid && !self.config.allow_invalid_tasks {
                    log::debug!("T{}\tNext active task {:?} is invalid and that is not allowed, aborting expansion", next_active_task.end, next_active_task.task);
                    return TreePolicyOutcome::NoValidTask(depth, path);
                }
                // Execute the task which finishes in the next node
                let after_next_task = if is_task_valid {
                    let state_diff_mut = StateDiffRefMut::new(&self.initial_state, &mut diff);
                    next_active_task.task.execute(
                        next_active_task.end,
                        state_diff_mut,
                        next_active_task.agent,
                    )
                } else {
                    None
                };

                // If we do not have a forced follow-up task...
                let after_next_task = if after_next_task.is_none() {
                    // And we have a forced planning task, handle it
                    if let Some(planning_task_duration) = self.config.planning_task_duration {
                        if next_active_task
                            .task
                            .downcast_ref::<PlanningTask>()
                            .is_none()
                        {
                            // the incoming task was not planning, so the next one should be
                            let task: Box<dyn Task<D>> =
                                Box::new(PlanningTask(planning_task_duration));
                            Some(task)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    after_next_task
                };

                // Create expanded node state
                // let was_planning = task.downcast_ref::<Plan>().is_some();
                let child_state = NodeInner::new(
                    &self.initial_state,
                    self.start_tick,
                    diff,
                    next_active_task.agent,
                    next_active_task.end,
                    child_tasks,
                );

                // Check if child node exists already
                let child_node =
                    if let Some((existing_node, _)) = self.nodes.get_key_value(&child_state) {
                        // Link existing child node
                        log::trace!("\tLinking to existing node {:?}", existing_node);
                        existing_node.clone()
                    } else {
                        // Create and insert new child node
                        log::trace!("\tCreating new node {:?}", child_state);
                        let child_node = Node::new(child_state);
                        self.nodes.insert(
                            child_node.clone(),
                            Edges::new(&child_node, &self.initial_state, after_next_task),
                        );
                        child_node
                    };

                // Create edge from parent to child
                let edge = new_edge(&node, &child_node, &agents);
                let edges = self.nodes.get_mut(&node).unwrap();
                edges.expanded_tasks.insert(task, edge.clone());

                // Push edge to path
                path.push(edge);

                depth += (child_node.tick - node.tick) as u32;
                log::debug!(
                    "T{}\tExpansion successful, node created with incoming task {:?}",
                    child_node.tick,
                    next_active_task.task
                );
                return TreePolicyOutcome::NodeCreated(depth, child_node, path);
            }

            // There is no child to this node, still return last node to ensure increase of visit count for this path
            if edges.child_visits() == 0 {
                log::debug!("T{}\tNode has no children, aborting expansion", node.tick);
                return TreePolicyOutcome::NoChildNode(depth, node, path);
            }

            // -------------------------
            // Selection
            // -------------------------
            // Node is fully expanded, perform selection
            let range = self.min_max_range(node.active_agent);
            let edges = self.nodes.get_mut(&node).unwrap();
            let task = edges
                .best_task(node.active_agent, self.config.exploration, range)
                .expect("No valid task!");
            log::trace!(
                "T{}\t{:?} - Select task: {:?}",
                node.tick,
                node.active_agent,
                task
            );
            let edge = edges.expanded_tasks.get(&task).unwrap().clone();

            // New node is the current child node
            let parent_tick = node.tick;
            node = {
                let edge = edge.lock().unwrap();
                edge.child()
            };
            let child_tick = node.tick;
            depth += (child_tick - parent_tick) as u32;

            // Push edge to path
            path.push(edge);
        }

        // We reached maximum depth, still return last node to ensure increase of visit count for this path
        log::debug!(
            "T{}\tReached maximum depth {}, aborting expansion",
            node.tick,
            depth
        );
        TreePolicyOutcome::DepthLimitReached(depth, node, path)
    }

    /// MCTS backpropagation phase. If rollout values are None, just increment the visits.
    fn backpropagation(
        &mut self,
        mut path: Vec<Edge<D>>,
        rollout_values: Option<BTreeMap<AgentId, f32>>,
    ) {
        // Backtracking
        path.drain(..).rev().for_each(|edge| {
            // Increment child node visit count
            let edge = &mut edge.lock().unwrap();
            edge.visits += 1;
            if let Some(rollout_values) = &rollout_values {
                let parent_node = edge.parent();
                let child_node = edge.child();
                let visits = edge.visits;
                let child_edges = self.nodes.get(&child_node).unwrap();

                let discount_factor =
                    Self::discount_factor(child_node.tick - parent_node.tick, &self.config);

                // Iterate all agents on edge
                edge.q_values.iter_mut().for_each(|(&agent, q_value_ref)| {
                    let parent_current_value =
                        parent_node.current_value_or_compute(agent, &self.initial_state);
                    let child_current_value =
                        child_node.current_value_or_compute(agent, &self.initial_state);

                    // Get q value from child, or rollout value if leaf node, or 0 if not in rollout
                    let mut child_q_value =
                        if let Some(value) = child_edges.q_value((visits, *q_value_ref), agent) {
                            value
                        } else {
                            rollout_values.get(&agent).copied().unwrap_or_default()
                        };

                    // Apply discount, there is no risk of double-discounting as if the parent and the child node
                    // have the same tick, the discount value will be 1.0
                    child_q_value *= discount_factor;

                    // Use Bellman Equation
                    let q_value = child_current_value - parent_current_value + child_q_value;

                    // Update q value for edge
                    *q_value_ref = *q_value;

                    // Update global q value range for agent
                    let q_value_range = self
                        .q_value_ranges
                        .entry(parent_node.active_agent)
                        .or_insert_with(|| Range {
                            start: VALUE_INFINITE,
                            end: VALUE_NEG_INFINITE,
                        });
                    q_value_range.start = q_value_range.start.min(q_value);
                    q_value_range.end = q_value_range.end.max(q_value);
                });
            }
        });
    }

    /// Calculates the discount factor for the tick duration.
    ///
    /// This basically calculates a half-life decay factor for the given duration.
    /// This means the discount factor will be 0.5 if the given ticks are equal to the configured half-life in the MCTS.
    fn discount_factor(duration: u64, config: &MCTSConfiguration) -> f32 {
        2f64.powf((-(duration as f64)) / (config.discount_hl as f64)) as f32
    }

    /// Returns the initial state at the root of the planning tree.
    pub fn initial_state(&self) -> &D::State {
        &self.initial_state
    }

    /// Returns the tick at the root of the planning tree.
    pub fn start_tick(&self) -> u64 {
        self.start_tick
    }

    /// Returns the agent the tree searches for.
    pub fn agent(&self) -> AgentId {
        self.root_agent
    }

    /// Returns the range of minimum and maximum global values.
    pub fn min_max_range(&self, agent: AgentId) -> Range<AgentValue> {
        self.q_value_ranges.get(&agent).cloned().unwrap_or(Range {
            start: VALUE_ZERO,
            end: VALUE_ZERO,
        })
    }

    /// Returns an iterator over all nodes and edges in the tree.
    pub fn nodes(&self) -> impl Iterator<Item = (&Node<D>, &Edges<D>)> {
        self.nodes.iter()
    }

    /// Returns the root node of the search tree.
    pub fn root_node(&self) -> Node<D> {
        self.root.clone()
    }

    /// Returns the edges associated to a given node.
    pub fn get_edges(&self, node: &Node<D>) -> Option<&Edges<D>> {
        self.nodes.get(node)
    }

    /// Returns the seed of the tree.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of nodes.
    pub fn edge_count(&self) -> usize {
        self.nodes
            .values()
            .map(|edges| edges.expanded_tasks.len())
            .sum()
    }

    /// Returns the duration of the last run.
    pub fn time(&self) -> Duration {
        self.time
    }

    /// Returns an estimation of the memory footprint of the MCTS struct.
    pub fn size(&self, task_size: fn(&dyn Task<D>) -> usize) -> usize {
        let mut size = 0;

        size += mem::size_of::<Self>();

        for (node, edges) in &self.nodes {
            size += node.size(task_size);
            size += edges.size(task_size);
        }

        size += self.q_value_ranges.len() * mem::size_of::<(AgentId, Range<f32>)>();

        size
    }
}

/// MCTS default policy using simulation-based rollout.
pub struct DefaultPolicyEstimator {}
impl<D: Domain> StateValueEstimator<D> for DefaultPolicyEstimator {
    fn estimate(
        &mut self,
        rng: &mut ChaCha8Rng,
        config: &MCTSConfiguration,
        initial_state: &D::State,
        start_tick: u64,
        node: &Node<D>,
        edges: &Edges<D>,
        depth: u32,
    ) -> Option<BTreeMap<AgentId, f32>> {
        let mut diff = node.diff.clone();
        log::debug!(
            "T{}\tStarting rollout with cur. values: {:?}",
            node.tick,
            node.current_values()
        );

        // In this map we collect at the same time both:
        // - the current value (measured from state and replaced in the course of simulation)
        // - the Q value (initially 0, updated in the course of simulation)
        let mut values: BTreeMap<AgentId, (AgentValue, f32)> = node
            .current_values()
            .iter()
            .map(|(&agent, &current_value)| (agent, (current_value, 0f32)))
            .collect::<BTreeMap<_, _>>();

        // Clone active tasks for child node, removing task of active agent
        let mut tasks = node
            .tasks
            .iter()
            .filter(|task| task.agent != node.active_agent)
            .cloned()
            .collect::<BTreeSet<_>>();

        // Sample a task for the node's unexpanded list, and put it in the queue
        let task = {
            if let Some((weights, tasks)) = edges.unexpanded_tasks.as_ref() {
                // Select task randomly
                let idx = weights.sample(rng);
                tasks[idx].clone()
            } else {
                // No unexpanded edges, q values are 0
                log::debug!(
                    "T{}\tNo unexpanded edges in node passed to rollout",
                    node.tick
                );
                return None;
            }
        };
        let new_active_task = ActiveTask::new(
            node.active_agent,
            task,
            node.tick,
            StateDiffRef::new(initial_state, &diff),
        );
        tasks.insert(new_active_task);
        let mut agents_with_tasks = tasks
            .iter()
            .map(|task| task.agent)
            .collect::<HashSet<AgentId>>();
        let mut agents = agents_with_tasks.iter().copied().collect();

        // Create the state we need to perform the simulation
        let rollout_start_tick = node.tick;
        let mut tick = node.tick;
        let mut depth = depth;
        while depth < config.depth {
            let state_diff = StateDiffRef::new(initial_state, &diff);

            // If there is no more task to do, return what we have so far
            if tasks.is_empty() {
                log::debug!(
                    "! T{} No more task to do in state\n{}",
                    tick,
                    D::get_state_description(state_diff)
                );
                break;
            }

            // Pop first task that is completed
            let active_task = tasks.iter().next().unwrap().clone();
            tasks.remove(&active_task);
            let active_agent = active_task.agent;
            agents_with_tasks.remove(&active_agent);

            // Compute elapsed time and update tick
            let elapsed = active_task.end - tick;
            tick = active_task.end;

            // If task is invalid, stop rollout
            let is_task_valid = active_task.task.is_valid(tick, state_diff, active_agent);
            if !is_task_valid && !config.allow_invalid_tasks {
                log::debug!(
                    "! T{} Not allowed invalid task {:?} by {:?} in state\n{}",
                    tick,
                    active_task.task,
                    active_agent,
                    D::get_state_description(state_diff)
                );
                break;
            } else if is_task_valid {
                log::trace!(
                    "✓ T{} Valid task {:?} by {:?} in state\n{}",
                    tick,
                    active_task.task,
                    active_agent,
                    D::get_state_description(state_diff)
                );
            } else {
                log::trace!(
                    "✓ T{} Skipping invalid task {:?} by {:?} in state\n{}",
                    tick,
                    active_task.task,
                    active_agent,
                    D::get_state_description(state_diff)
                );
            }

            // Execute the task
            let new_task = if is_task_valid {
                let state_diff_mut = StateDiffRefMut::new(initial_state, &mut diff);
                active_task.task.execute(tick, state_diff_mut, active_agent)
            } else {
                None
            };
            let new_state_diff = StateDiffRef::new(initial_state, &diff);

            // If we do not have a forced follow-up task...
            let new_task = if new_task.is_none() {
                // And we have a forced planning task, handle it
                if let Some(planning_task_duration) = config.planning_task_duration {
                    if active_task.task.downcast_ref::<PlanningTask>().is_none() {
                        // the incoming task was not planning, so the next one should be
                        let task: Box<dyn Task<D>> = Box::new(PlanningTask(planning_task_duration));
                        Some(task)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                new_task
            };

            // if the values for the agent executing the task are being tracked, update them
            if let Entry::Occupied(mut entry) = values.entry(active_agent) {
                let (current_value, estimated_value) = entry.get_mut();

                // Compute discount
                let discount =
                    MCTS::<D>::discount_factor(active_task.end - rollout_start_tick, config);

                // Update estimated value with discounted difference in current values
                let new_current_value = D::get_current_value(tick, new_state_diff, active_agent);
                *estimated_value += *(new_current_value - *current_value) * discount;
                *current_value = new_current_value;
            }

            // Update the list of tasks, only considering visible agents,
            // excluding the active agent (a new task for it will be added later)
            D::update_visible_agents(start_tick, tick, new_state_diff, active_agent, &mut agents);
            for agent in agents.iter() {
                if *agent != active_agent && !agents_with_tasks.contains(agent) {
                    tasks.insert(ActiveTask::new_idle(tick, *agent, active_agent));
                    agents_with_tasks.insert(*agent);
                }
            }

            // If active agent is visible, insert its next task, otherwise we forget about it
            if agents.contains(&active_agent) {
                // If no new task is available, select one randomly
                let new_task = new_task.or_else(|| {
                    // Get possible tasks
                    let tasks = D::get_tasks(tick, new_state_diff, active_agent);
                    if tasks.is_empty() {
                        return None;
                    }
                    // Safety-check that all tasks are valid
                    for task in &tasks {
                        debug_assert!(task.is_valid(tick, new_state_diff, active_agent));
                    }
                    // Get the weight for each task
                    let weights = WeightedIndex::new(
                        tasks
                            .iter()
                            .map(|task| task.weight(tick, new_state_diff, active_agent)),
                    )
                    .unwrap();
                    // Select task randomly
                    let idx = weights.sample(rng);
                    Some(tasks[idx].clone())
                });

                // If still none is available, stop caring about this agent
                if let Some(new_task) = new_task {
                    // Insert new task
                    let new_active_task = ActiveTask::new(
                        active_agent,
                        new_task,
                        tick,
                        StateDiffRef::new(initial_state, &diff),
                    );
                    tasks.insert(new_active_task);
                    agents_with_tasks.insert(active_agent);
                }
            }

            // Make sure we do not keep track of the agents outside of the horizon
            if agents_with_tasks.len() > agents.len() {
                agents_with_tasks.retain(|id| agents.contains(id));
            }

            // Update depth
            depth += elapsed as u32;
        }

        let q_values = values
            .iter()
            .map(|(agent, (_, q_value))| (*agent, *q_value))
            .collect();

        log::debug!(
            "T{}\tRollout to T{}: q values: {:?}",
            node.tick,
            depth,
            q_values
        );

        Some(q_values)
    }
}

/// When `graphviz` feature is enabled, provides plotting of the search tree.
#[cfg(feature = "graphviz")]
pub mod graphviz {
    use super::*;
    use std::hash::{Hash, Hasher};
    use std::{
        borrow::Cow,
        io::{self, Write},
        sync::{atomic::AtomicUsize, Arc},
    };

    use dot::{Arrow, Edges, GraphWalk, Id, Kind, LabelText, Labeller, Nodes, Style};

    /// Renders the search tree as graphviz's dot format.
    pub fn plot_mcts_tree<D: Domain, W: Write>(mcts: &MCTS<D>, w: &mut W) -> io::Result<()> {
        dot::render(mcts, w)
    }

    fn agent_color_hsv(agent: AgentId) -> (f32, f32, f32) {
        use palette::IntoColor;
        let mut hasher = std::collections::hash_map::DefaultHasher::default();
        agent.0.hash(&mut hasher);
        let bytes: [u8; 8] = hasher.finish().to_ne_bytes();
        let (h, s, v) = palette::Srgb::from_components((bytes[5], bytes[6], bytes[7]))
            .into_format::<f32>()
            .into_hsv::<palette::encoding::Srgb>()
            .into_components();

        ((h.to_degrees() + 180.) / 360., s, v)
    }

    struct Edge<D: Domain> {
        parent: Node<D>,
        child: Node<D>,
        task: Box<dyn Task<D>>,
        best: bool,
        visits: usize,
        score: f32,
        uct: f32,
        uct_0: f32,
        reward: f32,
    }

    impl<D: Domain> Clone for Edge<D> {
        fn clone(&self) -> Self {
            Edge {
                parent: self.parent.clone(),
                child: self.child.clone(),
                task: self.task.box_clone(),
                best: self.best,
                visits: self.visits,
                score: self.score,
                uct: self.uct,
                uct_0: self.uct_0,
                reward: self.reward,
            }
        }
    }

    /// The depth of the graph to plot, in number of nodes.
    static GRAPH_OUTPUT_DEPTH: AtomicUsize = AtomicUsize::new(4);

    /// Sets the depth of the graph to plot, in number of nodes.
    pub fn set_graph_output_depth(depth: usize) {
        graphviz::GRAPH_OUTPUT_DEPTH.store(depth, std::sync::atomic::Ordering::Relaxed);
    }
    /// Gets the depth of the graph to plot, in number of nodes.
    pub fn get_graph_output_depth() -> usize {
        GRAPH_OUTPUT_DEPTH.load(std::sync::atomic::Ordering::Relaxed)
    }

    impl<D: Domain> MCTS<D> {
        fn add_relevant_nodes(
            &self,
            nodes: &mut SeededHashSet<Node<D>>,
            node: &Node<D>,
            depth: usize,
        ) {
            if depth >= GRAPH_OUTPUT_DEPTH.load(std::sync::atomic::Ordering::Relaxed) {
                return;
            }

            nodes.insert(node.clone());

            let edges = self.nodes.get(node).unwrap();
            for edge in edges.expanded_tasks.values() {
                if let Ok(edge) = edge.try_lock() {
                    // Prevent recursion
                    if let Some(child) = edge.child.upgrade() {
                        // TODO: Priority queue
                        self.add_relevant_nodes(nodes, &child, depth + 1);
                    }
                }
            }
        }
    }

    impl<'a, D: Domain> GraphWalk<'a, Node<D>, Edge<D>> for MCTS<D> {
        fn nodes(&'a self) -> Nodes<'a, Node<D>> {
            let mut nodes = SeededHashSet::default();
            self.add_relevant_nodes(&mut nodes, &self.root, 0);

            Nodes::Owned(nodes.iter().cloned().collect::<Vec<_>>())
        }

        fn edges(&'a self) -> Edges<'a, Edge<D>> {
            let mut nodes = SeededHashSet::default();
            self.add_relevant_nodes(&mut nodes, &self.root, 0);

            let mut edge_vec = Vec::new();
            nodes.iter().for_each(|node| {
                let edges = self.nodes.get(node).unwrap();

                if !edges.expanded_tasks.is_empty() {
                    let range = self.min_max_range(self.root_agent);
                    let best_task = edges
                        .best_task(node.active_agent, 0., range.clone())
                        .unwrap();
                    let visits = edges.child_visits();
                    edges.expanded_tasks.iter().for_each(|(obj, _edge)| {
                        let edge = _edge.lock().unwrap();

                        let parent = edge.parent();
                        let child = edge.child();

                        if nodes.contains(&child) {
                            let child_value = child.current_value(node.active_agent);
                            let parent_value = parent.current_value(node.active_agent);
                            let reward = child_value - parent_value;
                            edge_vec.push(Edge {
                                parent: edge.parent(),
                                child,
                                task: obj.clone(),
                                best: obj == &best_task,
                                visits: edge.visits,
                                score: edge.q_values.get(&node.active_agent).copied().unwrap_or(0.),
                                uct: edge.uct(
                                    node.active_agent,
                                    visits,
                                    self.config.exploration,
                                    range.clone(),
                                ),
                                uct_0: edge.uct(node.active_agent, visits, 0., range.clone()),
                                reward: *reward,
                            });
                        }
                    });
                }
            });

            Edges::Owned(edge_vec)
        }

        fn source(&'a self, edge: &Edge<D>) -> Node<D> {
            edge.parent.clone()
        }

        fn target(&'a self, edge: &Edge<D>) -> Node<D> {
            edge.child.clone()
        }
    }

    impl<'a, D: Domain> Labeller<'a, Node<D>, Edge<D>> for MCTS<D> {
        fn graph_id(&'a self) -> Id<'a> {
            Id::new(format!("agent_{}", self.root_agent.0)).unwrap()
        }

        fn node_id(&'a self, n: &Node<D>) -> Id<'a> {
            Id::new(format!("_{:p}", Arc::as_ptr(n))).unwrap()
        }

        fn node_label(&'a self, n: &Node<D>) -> LabelText<'a> {
            let edges = self.nodes.get(n).unwrap();
            let q_v = edges.q_value((0, 0.), n.active_agent);
            let state_diff = StateDiffRef::new(&self.initial_state, &n.diff);
            let mut state = D::get_state_description(state_diff);
            if !state.is_empty() {
                state = state.replace('\n', "<br/>");
                state = format!("<br/><font point-size='10'>{state}</font>");
            }
            LabelText::HtmlStr(Cow::Owned(format!(
                "Agent {}<br/>T: {}, Q: {}<br/>V: {:?}{state}",
                n.active_agent.0,
                n.tick,
                q_v.map(|q_v| format!("{:.2}", q_v))
                    .unwrap_or_else(|| "None".to_owned()),
                n.current_values()
                    .iter()
                    .map(|(agent, value)| { (agent.0, **value) })
                    .collect::<BTreeMap<_, _>>(),
            )))
        }

        fn node_style(&'a self, node: &Node<D>) -> Style {
            if *node == self.root {
                Style::Bold
            } else {
                Style::Filled
            }
        }

        fn node_color(&'a self, node: &Node<D>) -> Option<LabelText<'a>> {
            let root_visits = self.nodes.get(&self.root).unwrap().child_visits();
            let visits = self.nodes.get(node).unwrap().child_visits();

            if *node == self.root {
                Some(LabelText::LabelStr(Cow::Borrowed("red")))
            } else {
                let (h, s, _v) = agent_color_hsv(node.active_agent);
                // let saturation = 95 -  * 50.) as usize;
                Some(LabelText::LabelStr(Cow::Owned(format!(
                    "{:.3} {:.3} 1.000",
                    h,
                    s * (visits as f32 / root_visits as f32).min(1.0)
                ))))
            }
        }

        fn edge_style(&'a self, edge: &Edge<D>) -> Style {
            if edge.best {
                Style::Bold
            } else {
                Style::Solid
            }
        }

        fn edge_color(&'a self, edge: &Edge<D>) -> Option<LabelText<'a>> {
            if edge.best {
                Some(LabelText::LabelStr(Cow::Borrowed("red")))
            } else {
                None
            }
        }

        fn edge_label(&'a self, edge: &Edge<D>) -> LabelText<'a> {
            LabelText::LabelStr(Cow::Owned(format!(
                "{:?}\nN: {}, R: {:.2}, Q: {:.2}\nU: {:.2} ({:.2} + {:.2})",
                edge.task.display_action(),
                edge.visits,
                edge.reward,
                edge.score,
                edge.uct,
                edge.uct_0,
                edge.uct - edge.uct_0
            )))
        }

        fn edge_start_arrow(&'a self, _e: &Edge<D>) -> Arrow {
            Arrow::none()
        }

        fn edge_end_arrow(&'a self, _e: &Edge<D>) -> Arrow {
            Arrow::normal()
        }

        fn kind(&self) -> Kind {
            Kind::Digraph
        }
    }
}
