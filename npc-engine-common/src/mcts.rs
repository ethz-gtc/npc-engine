use std::{collections::{BTreeMap, BTreeSet}};
use std::f32;
use std::hash::{Hash, Hasher};
use std::ops::{Range};
use std::time::{Duration, Instant};
use std::mem;

use rand::{prelude::{thread_rng, Distribution, RngCore, SeedableRng}, distributions::WeightedIndex};
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

/// The state of a running planner instance
pub struct MCTS<D: Domain> {
    // Statistics
    time: Duration,

    // Config
    pub(crate) config: MCTSConfiguration,
    state_value_estimator: Box<dyn StateValueEstimator<D>>,
    early_stop_condition: Option<Box<EarlyStopCondition>>,

    // Run-specific parameters
    pub(crate) root_agent: AgentId,
    seed: u64,

    // Nodes
    pub root: Node<D>,
    nodes: SeededHashMap<Node<D>, Edges<D>>,

    // Globals
    q_value_ranges: BTreeMap<AgentId, Range<AgentValue>>,

    // State before planning
    pub initial_state: D::State,
    pub start_tick: u64,

    // Rng
    rng: ChaCha8Rng,
}

/// Possible outcomes from a tree policy pass
enum TreePolicyOutcome<D: Domain> {
    NodeCreated(u32, Node<D>, Vec<Edge<D>>), // depth, new node, path
    NoValidTask(u32, Vec<Edge<D>>), // depth, path
    ActiveAgentNotInHorizon(u32, Vec<Edge<D>>), // depth, path
    NoChildNode(u32, Node<D>, Vec<Edge<D>>), // depth, node, path
    DepthLimitReached(u32, Node<D>, Vec<Edge<D>>), // depth, new node, path
}

impl<D: Domain> MCTS<D> {
    /// Instantiate a new search tree for the given state, with idle tasks for all agents and starting at tick 0
    pub fn new(
        initial_state: D::State,
        root_agent: AgentId,
        config: MCTSConfiguration,
    ) -> Self {
        Self::new_with_tasks(initial_state, root_agent, 0, Default::default(), config)
    }

    /// Instantiate a new search tree for the given state, with active tasks for all agents and starting at a given tick
    pub fn new_with_tasks(
        initial_state: D::State,
        root_agent: AgentId,
        start_tick: u64,
        tasks: ActiveTasks<D>,
        config: MCTSConfiguration,
    ) -> Self {
        // Create new root node
        let root = Node::new(NodeInner::new(
            &initial_state,
            Default::default(),
            root_agent,
            start_tick,
            tasks,
        ).expect("root_agent is not in the list of initial agents"));

        // Prepare nodes, reserve the maximum amount we could need
        let mut nodes = SeededHashMap::with_capacity_and_hasher(
            config.visits as usize + 1,
            SeededRandomState::default()
        );

        // Insert new root node
        let root_edges = Edges::new(&root, &initial_state, None);
        nodes.insert(root.clone(), root_edges);

        // Compute seed
        let cur_seed = config.seed.unwrap_or_else(|| thread_rng().next_u64());

        MCTS {
            time: Duration::default(),
            config,
            state_value_estimator: Box::new(DefaultPolicyEstimator {}),
            early_stop_condition: None,
            seed: cur_seed,
            root_agent,
            root,
            nodes,
            q_value_ranges: Default::default(),
            initial_state,
            start_tick,
            rng: ChaCha8Rng::seed_from_u64(cur_seed)
        }
    }

    /// Return best task, using exploration value of 0
    pub fn best_task_at_root(&self) -> Box<dyn Task<D>> {
        let range = self.min_max_range(self.root_agent);
        let edges = self.nodes.get(&self.root).unwrap();
        edges
            .best_task(self.root_agent, 0., range)
            .expect("No valid task!")
            .clone()
    }

    /// Execute the MCTS search. Returns the current best task.
    pub fn run(&mut self) -> Box<dyn Task<D>> {
        // Reset globals
        self.q_value_ranges.clear();

        let start = Instant::now();
        let max_visits = self.config.visits;
        for _ in 0..max_visits {
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
                        &leaf,
                        edges,
                        depth,
                        self.root_agent
                    );
                    (path, rollout_values)
                },
                TreePolicyOutcome::NoValidTask(_, path) => (path, None),
                TreePolicyOutcome::ActiveAgentNotInHorizon(_, path) => (path, None),
                TreePolicyOutcome::NoChildNode(_, _, path) => (path, None),
                TreePolicyOutcome::DepthLimitReached(_, _, path) => (path, None),
            };

            // Backpropagate results
            self.backpropagation(path, rollout_values);

            // Early stopping if told so by some user-defined condition
            if let Some(early_stop_condition) = &self.early_stop_condition {
                if early_stop_condition() {
                    break;
                }
            }
        }
        self.time = start.elapsed();

        self.best_task_at_root()
    }

    /// MCTS tree policy. Executes the `selection` and `expansion` phases.
    fn tree_policy(
        &mut self,
    ) -> TreePolicyOutcome<D> {
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
                log::debug!("T{}\t{:?} - Expand task: {:?}", node.tick, node.active_agent, task);

                // Set weight of chosen task to zero to mark it as expanded.
                // As updating weights returns an error if all weights are zero,
                // we have to handle this by setting unexpanded_tasks to None if we get an error.
                if weights.update_weights(&[(idx, &0.)]).is_err() {
                    // All weights being zero implies the node is fully expanded
                    edges.unexpanded_tasks = None;
                }

                // Clone active tasks for child node, removing task of active agent
                let mut child_tasks = node.tasks.iter()
                    .filter(|task| task.agent != node.active_agent)
                    .cloned()
                    .collect::<BTreeSet<_>>();
                // Create and insert new active task for the active agent and the selected task
                let active_task = ActiveTask::new(node.active_agent, task.clone(), node.tick, state_diff);
                child_tasks.insert(active_task);
                log::trace!("\tActive Tasks ({}):", child_tasks.len());
                for active_task in &child_tasks {
                    log::trace!("\t  {:?}: {:?} ends T{}", active_task.agent, active_task.task, active_task.end);
                }

                // Get task that finishes in the next node
                let next_active_task = child_tasks.iter().next().unwrap().clone();
                log::trace!("\tNext Active Task: {:?}: {:?} ends T{}", next_active_task.agent, next_active_task.task, next_active_task.end);

                // If it is not valid, abort this expansion
                if !next_active_task.task.is_valid(next_active_task.end, state_diff, next_active_task.agent) {
                    log::debug!("T{}\tNext active task {:?} is invalid, aborting expansion", next_active_task.end, next_active_task.task);
                    return TreePolicyOutcome::NoValidTask(depth, path);
                }
                // Execute the task which finishes in the next node
                let state_diff_mut = StateDiffRefMut::new(&self.initial_state, &mut diff);
                let after_next_task = next_active_task.task.execute(next_active_task.end, state_diff_mut, next_active_task.agent);

                // Create expanded node state
                let child_state = NodeInner::new(
                    &self.initial_state,
                    diff,
                    next_active_task.agent,
                    next_active_task.end,
                    child_tasks
                );

                // If no agents were found for this state, we cannot expand node
                match child_state {
                    None => {
                        log::debug!("T{}\tActive agent not in horizon after task {:?}, aborting expansion", next_active_task.end, next_active_task.task);
                        return TreePolicyOutcome::ActiveAgentNotInHorizon(depth, path);
                    },
                    Some(child_state) => {
                        // Check if child node exists already
                        let child_node = if let Some((existing_node, _)) = self.nodes.get_key_value(&child_state)
                        {
                            // Link existing child node
                            log::trace!("\tLinking to existing node {:?}", existing_node);
                            existing_node.clone()
                        } else {
                            // Create and insert new child node
                            log::trace!("\tCreating new node {:?}", child_state);
                            let child_node = Node::new(child_state);
                            self.nodes.insert(
                                child_node.clone(),
                                Edges::new(&child_node, &self.initial_state, after_next_task)
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
                        log::debug!("T{}\tExpansion successful, node created with incoming task {:?}", child_node.tick, next_active_task.task);
                        return TreePolicyOutcome::NodeCreated(depth, child_node, path);
                    }
                }
            }

            // There is no child to this node, still return last node to ensure increase of visit count for this path
            if edges.child_visits() == 0 {
                log::debug!("T{}\tNode has no children, aborting expansion", node.tick);
                return TreePolicyOutcome::NoChildNode(depth, node, path)
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
            log::trace!("T{}\t{:?} - Select task: {:?}", node.tick, node.active_agent, task);
            let edge = edges.expanded_tasks.get(&task).unwrap().clone();

            // New node is the current child node
            let parent_tick = node.tick;
            node = {
                let edge = edge.borrow();
                edge.child.upgrade().unwrap()
            };
            let child_tick = node.tick;
            depth += (child_tick - parent_tick) as u32;

            // Push edge to path
            path.push(edge);
        }

        // We reached maximum depth, still return last node to ensure increase of visit count for this path
        log::debug!("T{}\tReached maximum depth {}, aborting expansion", node.tick, depth);
        TreePolicyOutcome::DepthLimitReached(depth, node, path)
    }

    /// MCTS backpropagation phase. If rollout values are None, just increment the visits
    fn backpropagation(&mut self, mut path: Vec<Edge<D>>, rollout_values: Option<BTreeMap<AgentId, f32>>) {
        // Backtracking
        path.drain(..).rev().for_each(|edge| {
            // Increment child node visit count
            let edge = &mut edge.borrow_mut();
            edge.visits += 1;
            if let Some(rollout_values) = &rollout_values {
                let parent_node = edge.parent.upgrade().unwrap();
                let child_node = edge.child.upgrade().unwrap();
                let visits = edge.visits;
                let child_edges = self.nodes.get(&child_node).unwrap();

                let discount_factor = Self::discount_factor(child_node.tick - parent_node.tick, &self.config);

                // Iterate all agents on edge
                edge.q_values.iter_mut().for_each(|(&agent, q_value_ref)| {
                    let parent_current_value = parent_node.current_value_or_compute(agent, &self.initial_state);
                    let child_current_value = child_node.current_value_or_compute(agent, &self.initial_state);

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
                    let q_value_range = self.q_value_ranges
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
    /// This basically calculates a half-life decay factor for the given duration.
    /// This means the discount factor will be 0.5 if the given ticks are equal to the configured half-life in the MCTS.
    fn discount_factor(duration: u64, config: &MCTSConfiguration) -> f32 {
        2f64.powf((-(duration as f64)) / (config.discount_hl as f64)) as f32
    }

    // Returns the agent the tree searches for.
    pub fn agent(&self) -> AgentId {
        self.root_agent
    }

    // Returns the range of minimum and maximum global values.
    fn min_max_range(&self, agent: AgentId) -> Range<AgentValue> {
        self.q_value_ranges
            .get(&agent)
            .cloned()
            .unwrap_or(Range { start: VALUE_ZERO, end: VALUE_ZERO })
    }

    // Returns iterator over all nodes and edges in the tree.
    pub fn nodes(&self) -> impl Iterator<Item = (&Node<D>, &Edges<D>)> {
        self.nodes.iter()
    }

    // Returns edges associated to a given node
    pub fn get_edges(&self, node: &Node<D>) -> Option<&Edges<D>> {
        self.nodes.get(node)
    }

    // Returns the edges associated with a given node.
    pub fn edges(&self, node: &Node<D>) -> Option<&Edges<D>> {
        self.nodes.get(node)
    }

    // Returns the seed of the tree.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    // Returns the number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    // Returns the number of nodes
    pub fn edge_count(&self) -> usize {
        self.nodes.values().map(|edges| edges.expanded_tasks.len()).sum()
    }

    // Returns the duration of the last run
    pub fn time(&self) -> Duration {
        self.time
    }

    // Returns the size of MCTS struct
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

struct DefaultPolicyEstimator {}
impl<D: Domain> StateValueEstimator<D> for DefaultPolicyEstimator {
    /// MCTS default policy. Performs the simulation phase.
    fn estimate(
        &mut self,
        rng: &mut ChaCha8Rng,
        config: &MCTSConfiguration,
        initial_state: &D::State,
        node: &Node<D>,
        edges: &Edges<D>,
        depth: u32,
        _root_agent: AgentId,
    ) -> Option<BTreeMap<AgentId, f32>> {
        let mut diff = node.diff.clone();
        log::debug!("T{}\tStarting rollout with cur. values: {:?}", node.tick, node.current_values());

        // In this map we collect at the same time both:
        // - the current value (measured from state and replaced in the course of simulation)
        // - the Q value (initially 0, updated in the course of simulation)
        let mut values: BTreeMap<AgentId, (AgentValue, f32)> = node
            .current_values()
            .iter()
            .map(|(&agent, &current_value)|
                (agent, (current_value, 0f32))
            )
            .collect::<BTreeMap<_, _>>();

        // Clone active tasks for child node, removing task of active agent
        let mut tasks = node.tasks.iter()
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
                log::debug!("T{}\tNo unexpanded edges in node passed to rollout", node.tick);
                return None;
            }
        };
        let new_active_task = ActiveTask::new(
            node.active_agent,
            task,
            node.tick,
            StateDiffRef::new(initial_state, &diff)
        );
        tasks.insert(new_active_task);

        // Create the state we need to perform the simulation
        let start_tick = node.tick;
        let mut tick = node.tick;
        let mut depth = depth;
        while depth < config.depth {
            let state_diff = StateDiffRef::new(initial_state, &diff);

            // If there is no more task to do, return what we have so far
            if tasks.is_empty() {
                log::debug!("! T{} No more task to do in state\n{}",
                    tick,
                    D::get_state_description(state_diff)
                );
                break;
            }

            // Pop first task that is complete
            let active_task = tasks.iter().next().unwrap().clone();
            tasks.remove(&active_task);
            let agent = active_task.agent;

            // Lazily fetch current and estimated values for current agent, before this task completed
            let (current_value, estimated_value) = values
                .entry(active_task.agent)
                .or_insert_with(||
                    (
                        D::get_current_value(tick, state_diff, agent),
                        0f32
                    )
                );

            // Compute elapsed time and update tick
            let elapsed = active_task.end - tick;
            tick = active_task.end;

            // If task is invalid, stop rollout
            if !active_task.task.is_valid(tick, state_diff, agent) {
                log::debug!("! T{} Invalid task {:?} by {:?} in state\n{}",
                    tick, active_task.task, agent, D::get_state_description(state_diff)
                );
                break;
            } else {
                log::trace!("✓ T{} Valid task {:?} by {:?} in state\n{}",
                    tick, active_task.task, agent, D::get_state_description(state_diff)
                );
            }

            // Execute the task
            let state_diff_mut = StateDiffRefMut::new(initial_state, &mut diff);
            let new_task = active_task.task.execute(tick, state_diff_mut, agent);
            let new_state_diff = StateDiffRef::new(initial_state, &diff);

            // Compute discount
            let discount = MCTS::<D>::discount_factor(active_task.end - start_tick, config);

            // Update estimated value with discounted difference in current values
            let new_current_value = D::get_current_value(
                tick,
                new_state_diff,
                agent
            );
            *estimated_value += *(new_current_value - *current_value) * discount;
            *current_value = new_current_value;

            // If no new task is available, select one randomly
            let new_task = new_task.or_else(|| {
                // Get possible tasks
                let tasks = D::get_tasks(
                    tick,
                    new_state_diff,
                    agent
                );
                if tasks.is_empty() {
                    return None;
                }
                // Safety-check that all tasks are valid
                for task in &tasks {
                    debug_assert!(task.is_valid(tick, new_state_diff, agent));
                }
                // Get the weight for each task
                let weights =
                    WeightedIndex::new(tasks.iter().map(|task| {
                        task.weight(tick, new_state_diff, agent)
                    }))
                    .unwrap();
                // Select task randomly
                let idx = weights.sample(rng);
                Some(tasks[idx].clone())
            });

            // If still none is available, stop caring about this agent
            if let Some(new_task) = new_task {
                // Insert new task
                let new_active_task = ActiveTask::new(
                    agent,
                    new_task,
                    tick,
                    StateDiffRef::new(initial_state, &diff)
                );
                tasks.insert(new_active_task);
            }

            // Update depth
            depth += elapsed as u32;
        }

        let q_values = values
            .iter()
            .map(|(agent, (_, q_value))| (*agent, *q_value))
            .collect();

        log::debug!("T{}\tRollout to T{}: q values: {:?}", node.tick, depth, q_values);

        Some(q_values)
    }
}


#[cfg(feature = "graphviz")]
pub mod graphviz {
    use super::*;
    use std::{borrow::Cow, sync::{Arc, atomic::AtomicUsize}};

    use dot::{Arrow, Edges, GraphWalk, Id, Kind, LabelText, Labeller, Nodes, Style};

    pub fn agent_color_hsv(agent: AgentId) -> (f32, f32, f32) {
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

    pub struct Edge<D: Domain> {
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

    pub static GRAPH_OUTPUT_DEPTH: AtomicUsize = AtomicUsize::new(4);

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
                if let Ok(edge) = edge.try_borrow() {
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
                    let best_task = edges.best_task(node.active_agent, 0., range.clone()).unwrap();
                    let visits = edges.child_visits();
                    edges.expanded_tasks.iter().for_each(|(obj, _edge)| {
                        let edge = _edge.borrow();

                        let parent = edge.parent.upgrade().unwrap();
                        let child = edge.child.upgrade().unwrap();

                        if nodes.contains(&child) {
                            let child_value = child.current_value(node.active_agent);
                            let parent_value = parent.current_value(node.active_agent);
                            let reward = child_value - parent_value;
                            edge_vec.push(Edge {
                                parent: edge.parent.upgrade().unwrap(),
                                child,
                                task: obj.clone(),
                                best: obj == &best_task,
                                visits: edge.visits,
                                score: edge.q_values.get(&node.active_agent).copied().unwrap_or(0.),
                                uct: edge.uct(node.active_agent, visits, self.config.exploration, range.clone()),
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
            let v = edges.q_value((0, 0.), n.active_agent);
            let state_diff = StateDiffRef::new(&self.initial_state, &n.diff);
            let mut state = D::get_state_description(state_diff);
            if !state.is_empty() {
                state = state.replace("\n", "<br/>");
                state = format!("<br/><font point-size='10'>{state}</font>");
            }
            LabelText::HtmlStr(Cow::Owned(format!(
                "Agent {}<br/>Q: {}<br/>V: {:?}{state}",
                n.active_agent.0,
                v.map(|v| format!("{:.2}", v)).unwrap_or_else(|| "None".to_owned()),
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
