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

/// Estimator of state-value function: takes state of explored node and returns the estimated expected (discounted) values
pub trait StateValueEstimator<D: Domain> {
    fn estimate(
        &mut self,
        rnd: &mut ChaCha8Rng,
        config: &MCTSConfiguration,
        snapshot: &D::State,
        node: &Node<D>,
        depth: u32,
        agent: AgentId, // agent of the MCTS
    ) -> BTreeMap<AgentId, f32>;
}


/// The state of a running planner instance
pub struct MCTS<D: Domain> {
    // Statistics
    time: Duration,

    // Config
    pub(crate) config: MCTSConfiguration,
    state_value_estimator: Box<dyn StateValueEstimator<D>>,
    early_stop_condition: Option<Box<EarlyStopCondition>>,

    // Run-specific parameters
    pub(crate) agent: AgentId,
    seed: u64,

    // Nodes
    pub root: Node<D>,
    nodes: SeededHashMap<Node<D>, Edges<D>>,

    // Globals
    q_value_range: BTreeMap<AgentId, Range<f32>>,

    // State before planning
    pub initial_state: D::State,

    // Rng
    rng: ChaCha8Rng,
}

impl<D: Domain> MCTS<D> {
    /// Instantiate a new search tree for the given state.
    pub fn new(
        initial_state: D::State,
        agent: AgentId,
        _tasks: &BTreeMap<AgentId, Box<dyn Task<D>>>,
        config: MCTSConfiguration,
    ) -> Self {
        // Prepare nodes
        let mut nodes = SeededHashMap::with_capacity_and_hasher(
            config.visits as usize + 1, // FIXME: add unit test to know if this should be visits or visits + 1
            SeededRandomState::default()
        );

        // Create new root node
        let root = Node::new(NodeInner::new(
            &initial_state,
            Default::default(),
            agent,
            Default::default(),
        ));

        // Insert new root node
        nodes.insert(root.clone(), Edges::new(&root, &initial_state));

        // Compute seed
        let cur_seed = config.seed.unwrap_or_else(|| thread_rng().next_u64());

        MCTS {
            time: Duration::default(),
            config,
            state_value_estimator: Box::new(DefaultPolicyEstimator {}),
            early_stop_condition: None,
            seed: cur_seed,
            agent,
            root,
            nodes,
            q_value_range: Default::default(),
            initial_state,
            rng: ChaCha8Rng::seed_from_u64(cur_seed)
        }
    }

    /// Return best task, using exploration value of 0
    pub fn best_task_at_root(&self) -> Box<dyn Task<D>> {
        let range = self.min_max_range(self.agent);
        let edges = self.nodes.get(&self.root).unwrap();
        edges
            .best_task(self.agent, 0., range)
            .expect("No valid task!")
            .clone()
    }

    /// Execute the MCTS search. Returns the current best task.
    pub fn run(&mut self) -> Box<dyn Task<D>> {
        // Reset globals
        self.q_value_range.clear();

        let start = Instant::now();
        let max_visits = self.config.visits;
        for _ in 0..max_visits {
            // Execute tree policy
            let (depth, leaf, path) = self.tree_policy(self.root.clone());

            // Execute default policy
            let rollout_values = self.state_value_estimator.estimate(
                &mut self.rng,
                &self.config,
                &self.initial_state,
                &leaf,
                depth,
                self.agent
            );

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
        root: Node<D>,
    ) -> (u32, Node<D>, Vec<Edge<D>>) {
        // Find agents for current turn
        let agents = D::get_visible_agents(
            StateDiffRef::new(&self.initial_state, &root.diff),
            self.agent,
        );

        // Initial start agent is the current agent
        let start_agent = self.agent;

        let mut node = root.clone();

        // Maintain set of nodes seen to prevent cycles
        let mut seen_nodes = SeededHashSet::default();
        seen_nodes.insert(node.clone());

        // Path through the tree, including root and leaf
        let mut path = Vec::with_capacity(self.config.depth as usize * agents.len());

        // Execute selection until at most `max_depth`
        for depth in 0..self.config.depth {
            let mut iter = agents
                .iter()
                .chain(agents.iter())
                .skip_while(|agent| **agent != start_agent)
                .take(agents.len());

            // Iterator over each relevant agent, starting at the `start_agent`
            while let Some(&agent) = iter.next() {
                let range = self.min_max_range(node.agent);
                let nodes = &mut self.nodes;
                let mut edges = nodes.get_mut(&node).unwrap();

                {
                    let snapshot = &self.initial_state;

                    // If weights are non-empty, the node has not been fully expanded
                    if let Some((weights, tasks)) = edges.unexpanded_tasks.as_mut() {
                        let mut diff = node.diff.clone();

                        // Select task
                        let idx = weights.sample(&mut self.rng);
                        let task = tasks[idx].clone();
                        debug_assert!(task.is_valid(StateDiffRef::new(&self.initial_state, &diff), agent));
                        log::trace!("{:?} - Expand action: {}", agent, task);

                        // Updating weights returns an error if all weights are zero.
                        if weights.update_weights(&[(idx, &0.)]).is_err() {
                            // All weights being zero implies the node is fully expanded
                            edges.unexpanded_tasks = None;
                        }

                        let mut depth = depth;

                        // Get the next agent in the sequence
                        // If the iter wraps back to the starting agent, add one to the depth
                        let next_agent = iter.next().copied().unwrap_or_else(|| {
                            depth += 1;
                            start_agent
                        });

                        // Set new task for current agent, if one exists
                        let mut tasks = node.tasks.clone();
                        if let Some(next_task) =
                            task.execute(StateDiffRefMut::new(&self.initial_state, &mut diff), agent)
                        {
                            tasks.insert(agent, next_task);
                        } else {
                            tasks.remove(&agent);
                        }

                        // Create expanded node state
                        let child_state = NodeInner::new(snapshot, diff, next_agent, tasks);

                        // Check if child node exists already
                        let child_node = if let Some((existing_node, _)) =
                            nodes.get_key_value(&child_state)
                        {
                            // Link existing child node
                            existing_node.clone()
                        } else {
                            // Create and insert new child node
                            let child_node = Node::new(child_state);
                            nodes.insert(child_node.clone(), Edges::new(&child_node, &snapshot));
                            child_node
                        };

                        // Create edge from parent to child
                        let edge = new_edge(&node, &child_node, &agents);

                        let edges = nodes.get_mut(&node).unwrap();
                        edges.expanded_tasks.insert(task, edge.clone());

                        // Push edge to path
                        path.push(edge);

                        return (depth, child_node, path);
                    }
                }

                // Node is fully expanded, perform selection
                let task = edges
                    .best_task(node.agent, self.config.exploration, range)
                    .expect("No valid task!");
                log::trace!("{:?} - Select action: {}", agent, task);
                let edge = edges.expanded_tasks.get(&task).unwrap().clone();

                // New node is the current child node
                node = {
                    let edge = edge.borrow();
                    edge.child.upgrade().unwrap()
                };

                // Push edge to path
                path.push(edge);

                // If node is already seen, prevent cycle
                if !seen_nodes.insert(node.clone()) {
                    return (self.config.depth, node, path);
                }
            }

            /*
            We do not recalculate observed agents as this can lead to mismatching agent
            when expanding a node while the corresponding agent left the horizon.
            // Recalculate observed agents
            agents.clear();
            D::agents(
                SnapshotDiffRef::new(&self.snapshot, &node.diff),
                self.agent,
                &mut agents,
            );*/
        }

        (self.config.depth, node, path)
    }

    /// MCTS backpropagation phase.
    fn backpropagation(&mut self, mut path: Vec<Edge<D>>, rollout_values: BTreeMap<AgentId, f32>) {
        // Backtracking
        path.drain(..).rev().for_each(|edge| {
            // Increment child node visit count

            let edge = &mut edge.borrow_mut();
            edge.visits += 1;

            let parent = edge.parent.upgrade().unwrap();
            let child = edge.child.upgrade().unwrap();
            let visits = edge.visits;
            let child_edges = self.nodes.get(&child).unwrap();

            let q_values = &mut edge.q_values;
            let q_value_range = &mut self.q_value_range;
            let discount = self.config.discount;
            let snapshot = &self.initial_state;

            // Iterate all agents on edge
            q_values.iter_mut().for_each(|(&agent, q_value_ref)| {
                let parent_current_value = parent.current_values.get(&agent).copied().unwrap_or_else(|| {
                    D::get_current_value(
                        StateDiffRef::new(
                            snapshot,
                            parent.diff(),
                        ),
                        agent,
                    )
                });
                let child_current_value = child.current_values.get(&agent).copied().unwrap_or_else(|| {
                    D::get_current_value(
                        StateDiffRef::new(
                            snapshot,
                            child.diff(),
                        ),
                        agent,
                    )
                });

                // Get current value from child, or rollout value if leaf node
                let mut child_q_value =
                    if let Some(value) = child_edges.value((visits, *q_value_ref), agent) {
                        value
                    } else {
                        rollout_values.get(&agent).copied().unwrap_or_default()
                    };

                // Only discount once per agent per turn
                if agent == parent.agent {
                    child_q_value *= discount;
                }

                // Use Bellman Equation
                let q_value = child_current_value - parent_current_value + child_q_value;

                // Update q value for edge
                *q_value_ref = q_value;

                if agent == parent.agent {
                    // Update global score for agent
                    let global = q_value_range.entry(parent.agent).or_insert_with(|| Range {
                        start: f32::INFINITY,
                        end: f32::NEG_INFINITY,
                    });
                    global.start = global.start.min(q_value);
                    global.end = global.end.max(q_value);
                }
            });
        });
    }

    // Returns the agent the tree searches for.
    pub fn agent(&self) -> AgentId {
        self.agent
    }

    // Returns the range of minimum and maximum global values.
    fn min_max_range(&self, agent: AgentId) -> Range<f32> {
        self.q_value_range
            .get(&agent)
            .cloned()
            .unwrap_or(Range { start: 0., end: 0. })
    }

    // Returns iterator over all nodes and edges in the tree.
    pub fn nodes(&self) -> impl Iterator<Item = (&Node<D>, &Edges<D>)> {
        self.nodes.iter()
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

        size += self.q_value_range.len() * mem::size_of::<(AgentId, Range<f32>)>();

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
        snapshot: &D::State,
        node: &Node<D>,
        depth: u32,
        root_agent: AgentId,
    ) -> BTreeMap<AgentId, f32> {
        let mut start_agent = node.agent;
        let mut agents: BTreeSet<AgentId> = node.current_values
            .keys()
            .map(|agent| *agent)
            .collect()
        ;

        let mut diff = node.diff.clone();
        let mut task_map = node.tasks.clone();

        let mut num_rollouts = 0;

        // In this map we collect at the same time both:
        // - the current value (measured from state and replaced in the course of simulation)
        // - the Q value (initially 0, updated in the course of simulation)
        let mut values = node
            .current_values
            .iter()
            .map(|(&agent, &current_value)| (agent, (current_value, 0f32)))
            .collect::<BTreeMap<_, _>>();

        // Current discount multiplier
        let mut discount = 1.0;

        // Perform rollouts for remaining depth
        for rollout in 1..=(config.depth - depth) {
            let iter = agents
                .iter()
                .chain(agents.iter())
                .skip_while(|agent| **agent != start_agent)
                .enumerate()
                .take_while(|(i, agent)| *i == 0 || **agent != root_agent);

            // Iterator over each relevant agent, starting at the `start_agent`
            for (_, &agent) in iter {
                // Set current number of rollouts in case of early termination
                num_rollouts = rollout;

                // Lazily fetch current and estimated values for current agent
                let (current_value, estimated_value) = values.entry(agent).or_insert_with(|| {
                    (
                        D::get_current_value(StateDiffRef::new(snapshot, &diff), agent),
                        0f32
                    )
                });

                // Check task map for existing task
                let (tasks, weights) = match task_map.get(&agent) {
                    Some(task) if task.is_valid(StateDiffRef::new(snapshot, &diff), agent) => {
                        // Task exists, only option
                        (
                            vec![task.box_clone()],
                            WeightedIndex::new(&[1.]).ok()
                        )
                    }
                    _ => {
                        // No existing task, add all possible tasks
                        let tasks = D::get_tasks(StateDiffRef::new(snapshot, &diff), agent);
                        let weights_iter = tasks.iter().map(|task| {
                            task.weight(StateDiffRef::new(snapshot, &diff) as _, agent)
                        });
                        let weights = WeightedIndex::new(weights_iter).ok();
                        (
                            tasks,
                            weights
                        )
                    }
                };

                if let Some(mut weights) = weights {
                    // Get random task, assert it is valid
                    let mut idx;
                    let mut task;
                    while {
                        idx = weights.sample(rng);
                        task = &tasks[idx];
                        log::trace!("{:?} - Rollout: {}", agent, task);

                        !task.is_valid(StateDiffRef::new(snapshot, &diff), agent)
                    } {
                        weights
                            .update_weights(&[(idx, &0.)])
                            .expect("No valid actions!");
                    }

                    // Execute task for agent
                    if let Some(task) =
                        task.execute(StateDiffRefMut::new(snapshot, &mut diff), agent)
                    {
                        task_map.insert(agent, task);
                    } else {
                        task_map.remove(&agent);
                    }

                    // Update estimated value with discounted difference in current values
                    let new_current_value = D::get_current_value(StateDiffRef::new(snapshot, &diff), agent);
                    *estimated_value += (new_current_value - *current_value) * discount;
                    *current_value = new_current_value;
                } else {
                    break;
                };
            }

            // Recalculate agents
            agents.clear();
            D::update_visible_agents(StateDiffRef::new(snapshot, &diff), node.agent, &mut agents);

            // Iterator has been exu
            start_agent = root_agent;

            // Increment discount
            discount *= config.discount;
        }

        let current_values = node.current_values.clone();
        let q_values = values
            .iter()
            .map(|(agent, (_, value))| (*agent, *value))
            .collect();

        log::trace!("Rollout: {:?}, {:?}, {:?}", num_rollouts, current_values, q_values);

        q_values
    }
}


#[cfg(feature = "graphviz")]
mod graphviz {
    use super::*;
    use std::{borrow::Cow, sync::Arc};

    use dot::{Arrow, Edges, GraphWalk, Id, Kind, LabelText, Labeller, Nodes, Style};

    pub fn agent_color_hsv(agent: AgentId) -> (f32, f32, f32) {
        use palette::IntoColor;
        let mut hasher = std::collections::hash_map::DefaultHasher::default();
        agent.0.hash(&mut hasher);
        unsafe {
            let bytes: [u8; 8] = std::mem::transmute(hasher.finish());
            let (h, s, v) = palette::Srgb::from_components((bytes[5], bytes[6], bytes[7]))
                .into_format::<f32>()
                .into_hsv::<palette::encoding::Srgb>()
                .into_components();

            ((h.to_degrees() + 180.) / 360., s, v)
        }
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
                visits: self.visits.clone(),
                score: self.score,
                uct: self.uct,
                uct_0: self.uct_0,
                reward: self.reward,
            }
        }
    }

    impl<D: Domain> MCTS<D> {
        fn add_relevant_nodes(
            &self,
            nodes: &mut SeededHashSet<Node<D>>,
            node: &Node<D>,
            depth: usize,
        ) {
            if depth >= 3 {
                return;
            }

            nodes.insert(node.clone());

            let edges = self.nodes.get(node).unwrap();
            for (_task, edge) in &edges.expanded_tasks {
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
                    let range = self.min_max_range(self.agent);
                    let best_task = edges.best_task(node.agent, 0., range.clone()).unwrap();
                    let visits = edges.child_visits();
                    edges.expanded_tasks.iter().for_each(|(obj, _edge)| {
                        let edge = _edge.borrow();

                        let parent = edge.parent.upgrade().unwrap();
                        let child = edge.child.upgrade().unwrap();

                        if nodes.contains(&child) {
                            let reward = child
                                .current_values
                                .get(&node.agent)
                                .copied()
                                .unwrap_or_default()
                                - parent
                                    .current_values
                                    .get(&node.agent)
                                    .copied()
                                    .unwrap_or_default();
                            edge_vec.push(Edge {
                                parent: edge.parent.upgrade().unwrap(),
                                child,
                                task: obj.clone(),
                                best: obj == &best_task,
                                visits: edge.visits,
                                score: edge.q_values.get(&node.agent).copied().unwrap_or(0.),
                                uct: edge.uct(node.agent, visits, self.config.exploration, range.clone()),
                                uct_0: edge.uct(node.agent, visits, 0., range.clone()),
                                reward,
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
            Id::new(format!("agent_{}", self.agent.0)).unwrap()
        }

        fn node_id(&'a self, n: &Node<D>) -> Id<'a> {
            Id::new(format!("_{:p}", Arc::as_ptr(n))).unwrap()
        }

        fn node_label(&'a self, n: &Node<D>) -> LabelText<'a> {
            let edges = self.nodes.get(n).unwrap();
            let v = edges.value((0, 0.), n.agent);

            LabelText::LabelStr(Cow::Owned(format!(
                "Agent {}\nV: {}\nFitnesses: {:?}",
                n.agent.0,
                v.map(|v| format!("{:.2}", v)).unwrap_or("None".to_owned()),
                n.current_values
                    .iter()
                    .map(|(agent, value)| { (agent.0, *value) })
                    .collect::<SeededHashMap<_, _>>(),
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
                let (h, s, _v) = agent_color_hsv(node.agent);
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
                "{}\nN: {}, R: {:.2}, Q: {:.2}\nU: {:.2} ({:.2} + {:.2})",
                edge.task,
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

#[cfg(test)]
mod value_tests {
    use std::fmt;

    use super::*;

    use crate::{Behavior, Task};

    pub(crate) struct TestEngine;

    #[derive(Debug)]
    pub(crate) struct State(usize);

    #[derive(Debug, Default, Eq, Hash, Clone, PartialEq)]
    pub(crate) struct Diff(usize);

    impl Domain for TestEngine {
        type State = State;
        type Diff = Diff;
        type DisplayAction = ();

        fn list_behaviors() -> &'static [&'static dyn Behavior<Self>] {
            &[&TestBehavior]
        }

        fn get_current_value(state: StateDiffRef<Self>, _agent: AgentId) -> f32 {
            state.value()
        }

        fn update_visible_agents(_state: StateDiffRef<Self>, agent: AgentId, agents: &mut BTreeSet<AgentId>) {
            agents.insert(agent);
        }
    }

    trait TestState {
        fn value(&self) -> f32;
    }

    trait TestStateMut {
        fn increment(&mut self, agent: AgentId);
    }

    impl TestState for StateDiffRef<'_, TestEngine> {
        fn value(&self) -> f32 {
            self.initial_state.0 as f32 + self.diff.0 as f32
        }
    }

    impl TestStateMut for StateDiffRefMut<'_, TestEngine> {
        fn increment(&mut self, _agent: AgentId) {
            self.diff.0 += 1;
        }
    }

    #[derive(Copy, Clone, Debug)]
    struct TestBehavior;

    impl fmt::Display for TestBehavior {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "TestBehavior")
        }
    }

    impl Behavior<TestEngine> for TestBehavior {
        fn add_own_tasks(
            &self,
            _state: StateDiffRef<TestEngine>,
            _agent: AgentId,
            tasks: &mut Vec<Box<dyn Task<TestEngine>>>,
        ) {
            tasks.push(Box::new(TestTask) as _);
        }

        fn is_valid(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
            true
        }
    }

    #[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
    struct TestTask;

    impl fmt::Display for TestTask {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "TestTask")
        }
    }

    impl Task<TestEngine> for TestTask {
        fn weight(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> f32 {
            1.
        }

        fn is_valid(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
            true
        }

        fn execute(
            &self,
            mut state: StateDiffRefMut<TestEngine>,
            agent: AgentId,
        ) -> Option<Box<dyn Task<TestEngine>>> {
            state.increment(agent);
            None
        }

        fn display_action(&self) -> <value_tests::TestEngine as Domain>::DisplayAction {
            ()
        }

        fn box_eq(&self, other: &Box<dyn Task<TestEngine>>) -> bool {
            other
                .downcast_ref::<Self>()
                .map(|other| self.eq(other))
                .unwrap_or_default()
        }

        fn box_clone(&self) -> Box<dyn Task<TestEngine>> {
            Box::new(self.clone())
        }

        fn box_hash(&self, mut state: &mut dyn Hasher) {
            self.hash(&mut state)
        }
    }

    const EPSILON: f32 = 0.001;

    #[test]
    fn linear_bellman() {
        const CONFIG: MCTSConfiguration = MCTSConfiguration {
            visits: 10_000,
            depth: 5,
            exploration: 1.414,
            discount: 0.95,
            seed: None
        };
        let agent = AgentId(0);

        let world = State(0);
        let mut mcts = MCTS::<TestEngine>::new(
            world,
            agent,
            &BTreeMap::new(),
            CONFIG,
        );

        fn expected_value(discount: f32, depth: u32) -> f32 {
            let mut value = 0.;

            for _ in 0..depth {
                value = 1. + discount * value;
            }

            value
        }

        let task = mcts.run();
        assert!(task.downcast_ref::<TestTask>().is_some());
        // Check length is depth with root
        assert_eq!((CONFIG.depth + 1) as usize, mcts.nodes.len());

        let mut node = mcts.root;

        {
            assert_eq!(Diff(0), node.diff);
        }

        for i in 1..CONFIG.depth {
            let edges = mcts.nodes.get(&node).unwrap();
            assert_eq!(edges.expanded_tasks.len(), 1);
            let edge_rc = edges.expanded_tasks.values().nth(0).unwrap();
            let edge = edge_rc.borrow();

            node = edge.child.upgrade().unwrap();

            assert_eq!(Diff(i as usize), node.diff);
            assert_eq!((CONFIG.visits - i + 1) as usize, edge.visits);
            assert!(
                (expected_value(CONFIG.discount, CONFIG.depth - i + 1) - *edge.q_values.get(&agent).unwrap())
                    .abs()
                    < EPSILON
            );
        }
    }
}

#[cfg(test)]
mod branching_tests {
    use std::{ops::Range, fmt};

    use super::*;

    use crate::{Behavior, Task};

    struct TestEngine;

    #[derive(Debug)]
    struct State(usize);

    #[derive(Debug, Default, Eq, Hash, Clone, PartialEq)]
    struct Diff(usize);

    impl Domain for TestEngine {
        type State = State;
        type Diff = Diff;
        type DisplayAction = ();

        fn list_behaviors() -> &'static [&'static dyn Behavior<Self>] {
            &[&TestBehaviorA, &TestBehaviorB]
        }

        fn get_current_value(state: StateDiffRef<Self>, _agent: AgentId) -> f32 {
            state.value()
        }

        fn update_visible_agents(_state: StateDiffRef<Self>, agent: AgentId, agents: &mut BTreeSet<AgentId>) {
            agents.insert(agent);
        }
    }

    trait TestState {
        fn value(&self) -> f32;
    }

    trait TestStateMut {
        fn increment(&mut self, agent: AgentId);
    }

    impl TestState for StateDiffRef<'_, TestEngine> {
        fn value(&self) -> f32 {
            self.initial_state.0 as f32 + self.diff.0 as f32
        }
    }

    impl TestStateMut for StateDiffRefMut<'_, TestEngine> {
        fn increment(&mut self, _agent: AgentId) {
            self.diff.0 += 1;
        }
    }

    #[derive(Copy, Clone, Debug)]
    struct TestBehaviorA;

    impl fmt::Display for TestBehaviorA {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "TestBehaviorA")
        }
    }

    impl Behavior<TestEngine> for TestBehaviorA {
        fn add_own_tasks(
            &self,
            _state: StateDiffRef<TestEngine>,
            _agent: AgentId,
            tasks: &mut Vec<Box<dyn Task<TestEngine>>>,
        ) {
            tasks.push(Box::new(TestTask(true)) as _);
        }

        fn is_valid(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
            true
        }
    }

    #[derive(Copy, Clone, Debug)]
    struct TestBehaviorB;

    impl fmt::Display for TestBehaviorB {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "TestBehaviorB")
        }
    }

    impl Behavior<TestEngine> for TestBehaviorB {
        fn add_own_tasks(
            &self,
            _state: StateDiffRef<TestEngine>,
            _agent: AgentId,
            tasks: &mut Vec<Box<dyn Task<TestEngine>>>,
        ) {
            tasks.push(Box::new(TestTask(false)) as _);
        }

        fn is_valid(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
            true
        }
    }

    #[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
    struct TestTask(bool);

    impl fmt::Display for TestTask {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "TestTask({:?})", self.0)
        }
    }

    impl Task<TestEngine> for TestTask {
        fn weight(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> f32 {
            1.
        }

        fn is_valid(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
            true
        }

        fn execute(
            &self,
            mut state: StateDiffRefMut<TestEngine>,
            agent: AgentId,
        ) -> Option<Box<dyn Task<TestEngine>>> {
            state.increment(agent);
            None
        }

        fn display_action(&self) -> <value_tests::TestEngine as Domain>::DisplayAction {
            ()
        }

        fn box_eq(&self, other: &Box<dyn Task<TestEngine>>) -> bool {
            other
                .downcast_ref::<Self>()
                .map(|other| self.eq(other))
                .unwrap_or_default()
        }

        fn box_clone(&self) -> Box<dyn Task<TestEngine>> {
            Box::new(self.clone())
        }

        fn box_hash(&self, mut state: &mut dyn Hasher) {
            self.hash(&mut state)
        }
    }

    const EPSILON: f32 = 0.001;

    #[test]
    fn ucb() {
        const CONFIG: MCTSConfiguration = MCTSConfiguration {
            visits: 10,
            depth: 1,
            exploration: 1.414,
            discount: 0.95,
            seed: None
        };
        let agent = AgentId(0);

        let state = State(Default::default());
        let mut mcts = MCTS::<TestEngine>::new(
            state,
            agent,
            &BTreeMap::new(),
            CONFIG
        );

        let task = mcts.run();
        assert!(task.downcast_ref::<TestTask>().is_some());
        // Check length is depth with root
        assert_eq!((CONFIG.depth + 1) as usize, mcts.nodes.len());

        let node = mcts.root;
        let edges = mcts.nodes.get(&node).unwrap();
        let root_visits = edges.child_visits();

        let edge_a = edges
            .expanded_tasks
            .get(&(Box::new(TestTask(true)) as Box<dyn Task<TestEngine>>))
            .unwrap()
            .borrow();
        let edge_b = edges
            .expanded_tasks
            .get(&(Box::new(TestTask(false)) as Box<dyn Task<TestEngine>>))
            .unwrap()
            .borrow();

        assert!(
            (edge_a.uct(
                AgentId(0),
                root_visits,
                CONFIG.exploration,
                Range {
                    start: 0.0,
                    end: 1.0
                }
            ) - 1.9597)
                .abs()
                < EPSILON
        );
        assert!(
            (edge_b.uct(
                AgentId(0),
                root_visits,
                CONFIG.exploration,
                Range {
                    start: 0.0,
                    end: 1.0
                }
            ) - 1.9597)
                .abs()
                < EPSILON
        );
    }
}

#[cfg(test)]
mod seeding_tests {

    use std::fmt;

    use super::*;

    use crate::{Behavior, Task};

    struct TestEngine;

    #[derive(Debug, Clone, Copy)]
    struct State(usize);

    #[derive(Debug, Default, Eq, Hash, Clone, PartialEq)]
    struct Diff(usize);

    impl Domain for TestEngine {
        type State = State;
        type Diff = Diff;
        type DisplayAction = ();

        fn list_behaviors() -> &'static [&'static dyn Behavior<Self>] {
            &[&TestBehavior]
        }

        fn get_current_value(state: StateDiffRef<Self>, _agent: AgentId) -> f32 {
            state.value()
        }

        fn update_visible_agents(_state: StateDiffRef<Self>, agent: AgentId, agents: &mut BTreeSet<AgentId>) {
            agents.insert(agent);
        }
    }

    trait TestState {
        fn value(&self) -> f32;
    }

    trait TestStateMut {
        fn add_value(&mut self, agent: AgentId, amount: usize);
    }

    impl TestState for StateDiffRef<'_, TestEngine> {
        fn value(&self) -> f32 {
            self.initial_state.0 as f32 + self.diff.0 as f32
        }
    }

    impl TestStateMut for StateDiffRefMut<'_, TestEngine> {
        fn add_value(&mut self, _agent: AgentId, amount: usize) {
            self.diff.0 += amount;
        }
    }

    #[derive(Copy, Clone, Debug)]
    struct TestBehavior;

    impl fmt::Display for TestBehavior {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "TestBehavior")
        }
    }

    impl Behavior<TestEngine> for TestBehavior {
        fn add_own_tasks(
            &self,
            _state: StateDiffRef<TestEngine>,
            _agent: AgentId,
            tasks: &mut Vec<Box<dyn Task<TestEngine>>>,
        ) {
            for i in 0..10 {
                tasks.push(Box::new(TestTask(i)) as _);
            }
        }

        fn is_valid(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
            true
        }
    }

    #[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
    struct TestTask(usize);

    impl fmt::Display for TestTask {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "TestTask({:?})", self.0)
        }
    }

    impl Task<TestEngine> for TestTask {
        fn weight(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> f32 {
            1.
        }

        fn is_valid(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
            true
        }

        fn execute(
            &self,
            mut state: StateDiffRefMut<TestEngine>,
            agent: AgentId,
        ) -> Option<Box<dyn Task<TestEngine>>> {
            state.add_value(agent, self.0.min(1));
            None
        }

        fn display_action(&self) -> <value_tests::TestEngine as Domain>::DisplayAction {
            ()
        }

        fn box_eq(&self, other: &Box<dyn Task<TestEngine>>) -> bool {
            other
                .downcast_ref::<Self>()
                .map(|other| self.eq(other))
                .unwrap_or_default()
        }

        fn box_clone(&self) -> Box<dyn Task<TestEngine>> {
            Box::new(self.clone())
        }

        fn box_hash(&self, mut state: &mut dyn Hasher) {
            self.hash(&mut state)
        }
    }

    #[test]
    fn seed() {
        let agent = AgentId(0);
        for _ in 0..5 {
            let seed = thread_rng().next_u64();
            let config = MCTSConfiguration {
                visits: 1000,
                depth: 10,
                exploration: 1.414,
                discount: 0.95,
                seed: Some(seed),
            };
            let state = State(Default::default());
            let mut mcts = MCTS::<TestEngine>::new(
                state,
                agent,
                &BTreeMap::new(),
                config.clone(),
            );

            let result = mcts.run();

            for _ in 0..10 {
                let mut mcts = MCTS::<TestEngine>::new(
                    state,
                    agent,
                    &BTreeMap::new(),
                    config.clone(),
                );

                assert!(result == mcts.run());
            }
        }
    }
}

#[cfg(test)]
mod sanity_tests {
    use super::*;

    mod deferment {
        use std::fmt;

        use super::*;

        use crate::{Behavior, Task};

        struct TestEngine;

        #[derive(Debug)]
        struct State {
            value: isize,
            investment: isize,
        }

        #[derive(Debug, Default, Eq, Hash, Clone, PartialEq)]
        struct Diff {
            value: isize,
            investment: isize,
        }

        impl Domain for TestEngine {
            type State = State;
            type Diff = Diff;
            type DisplayAction = ();

            fn list_behaviors() -> &'static [&'static dyn Behavior<Self>] {
                &[&TestBehavior]
            }

            fn get_current_value(state: StateDiffRef<Self>, _agent: AgentId) -> f32 {
                state.value()
            }

            fn update_visible_agents(_state: StateDiffRef<Self>, agent: AgentId, agents: &mut BTreeSet<AgentId>) {
                agents.insert(agent);
            }
        }

        trait TestState {
            fn value(&self) -> f32;
        }

        trait TestStateMut {
            fn add_value(&mut self, amount: isize);
            fn add_investment(&mut self, amount: isize);
            fn redeem_deferred(&mut self);
        }

        impl TestState for StateDiffRef<'_, TestEngine> {
            fn value(&self) -> f32 {
                self.initial_state.value as f32 + self.diff.value as f32
            }
        }

        impl TestStateMut for StateDiffRefMut<'_, TestEngine> {
            fn add_value(&mut self, amount: isize) {
                self.diff.value += amount;
            }

            fn add_investment(&mut self, amount: isize) {
                self.diff.investment += amount;
            }

            fn redeem_deferred(&mut self) {
                self.diff.value += 3 * (self.initial_state.investment + self.diff.investment);
                self.diff.investment = 0 - self.initial_state.investment;
            }
        }

        #[derive(Copy, Clone, Debug)]
        struct TestBehavior;

        impl fmt::Display for TestBehavior {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "TestBehavior")
            }
        }

        impl Behavior<TestEngine> for TestBehavior {
            fn add_own_tasks(
                &self,
                _state: StateDiffRef<TestEngine>,
                _agent: AgentId,
                tasks: &mut Vec<Box<dyn Task<TestEngine>>>,
            ) {
                tasks.push(Box::new(TestTaskDirect) as _);
                tasks.push(Box::new(TestTaskDefer) as _);
            }

            fn is_valid(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
                true
            }
        }

        #[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
        struct TestTaskDirect;

        impl fmt::Display for TestTaskDirect {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "TestTaskDirect")
            }
        }

        impl Task<TestEngine> for TestTaskDirect {
            fn weight(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> f32 {
                1.
            }

            fn is_valid(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
                true
            }

            fn execute(
                &self,
                mut state: StateDiffRefMut<TestEngine>,
                _agent: AgentId,
            ) -> Option<Box<dyn Task<TestEngine>>> {
                state.redeem_deferred();
                state.add_value(1);
                None
            }

            fn display_action(&self) -> <value_tests::TestEngine as Domain>::DisplayAction {
                ()
            }

            fn box_eq(&self, other: &Box<dyn Task<TestEngine>>) -> bool {
                other
                    .downcast_ref::<Self>()
                    .map(|other| self.eq(other))
                    .unwrap_or_default()
            }

            fn box_clone(&self) -> Box<dyn Task<TestEngine>> {
                Box::new(self.clone())
            }

            fn box_hash(&self, mut state: &mut dyn Hasher) {
                self.hash(&mut state)
            }
        }

        #[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
        struct TestTaskDefer;

        impl fmt::Display for TestTaskDefer {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "TestTaskDefer")
            }
        }

        impl Task<TestEngine> for TestTaskDefer {
            fn weight(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> f32 {
                1.
            }

            fn is_valid(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
                true
            }

            fn execute(
                &self,
                mut state: StateDiffRefMut<TestEngine>,
                _agent: AgentId,
            ) -> Option<Box<dyn Task<TestEngine>>> {
                state.redeem_deferred();
                state.add_investment(1);
                None
            }

            fn display_action(&self) -> <value_tests::TestEngine as Domain>::DisplayAction {
                ()
            }

            fn box_eq(&self, other: &Box<dyn Task<TestEngine>>) -> bool {
                other
                    .downcast_ref::<Self>()
                    .map(|other| self.eq(other))
                    .unwrap_or_default()
            }

            fn box_clone(&self) -> Box<dyn Task<TestEngine>> {
                Box::new(self.clone())
            }

            fn box_hash(&self, mut state: &mut dyn Hasher) {
                self.hash(&mut state)
            }
        }

        #[test]
        fn deferment() {
            const CONFIG: MCTSConfiguration = MCTSConfiguration {
                visits: 1000,
                depth: 10,
                exploration: 1.414,
                discount: 0.95,
                seed: None
            };
            let agent = AgentId(0);

            let state = State {
                value: Default::default(),
                investment: Default::default(),
            };
            let mut mcts = MCTS::<TestEngine>::new(
                state,
                agent,
                &BTreeMap::new(),
                CONFIG
            );

            let task = mcts.run();
            assert!(task.downcast_ref::<TestTaskDefer>().is_some());
        }
    }

    mod negative {
        use std::fmt;

        use super::*;

        use crate::{Behavior, Task};

        struct TestEngine;

        #[derive(Debug)]
        struct State {
            value: isize,
        }

        #[derive(Debug, Default, Eq, Hash, Clone, PartialEq)]
        struct Diff {
            value: isize,
        }

        impl Domain for TestEngine {
            type State = State;
            type Diff = Diff;
            type DisplayAction = ();

            fn list_behaviors() -> &'static [&'static dyn Behavior<Self>] {
                &[&TestBehavior]
            }

            fn get_current_value(state: StateDiffRef<Self>, _agent: AgentId) -> f32 {
                state.value()
            }

            fn update_visible_agents(_state: StateDiffRef<Self>, agent: AgentId, agents: &mut BTreeSet<AgentId>) {
                agents.insert(agent);
            }

            fn get_visible_agents(state: StateDiffRef<Self>, agent: AgentId) -> BTreeSet<AgentId> {
                let mut agents = BTreeSet::new();
                Self::update_visible_agents(state, agent, &mut agents);
                agents
            }

            fn get_tasks<'a>(
                state: StateDiffRef<'a, Self>,
                agent: AgentId
            ) -> Vec<Box<dyn Task<Self>>> {
                let mut actions = Vec::new();
                Self::list_behaviors()
                    .iter()
                    .filter(|behavior| behavior.is_valid(state, agent))
                    .for_each(|behavior| behavior.add_tasks(state, agent, &mut actions));

                actions.dedup();
                actions
            }
        }

        trait TestState {
            fn value(&self) -> f32;
        }

        trait TestStateMut {
            fn add_value(&mut self, amount: isize);
        }

        impl TestState for StateDiffRef<'_, TestEngine> {
            fn value(&self) -> f32 {
                self.initial_state.value as f32 + self.diff.value as f32
            }
        }

        impl TestStateMut for StateDiffRefMut<'_, TestEngine> {
            fn add_value(&mut self, amount: isize) {
                self.diff.value += amount;
            }
        }

        #[derive(Copy, Clone, Debug)]
        struct TestBehavior;

        impl fmt::Display for TestBehavior {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "TestBehavior")
            }
        }

        impl Behavior<TestEngine> for TestBehavior {
            fn add_own_tasks(
                &self,
                _state: StateDiffRef<TestEngine>,
                _agent: AgentId,
                tasks: &mut Vec<Box<dyn Task<TestEngine>>>,
            ) {
                tasks.push(Box::new(TestTaskNoop) as _);
                tasks.push(Box::new(TestTaskNegative) as _);
            }

            fn is_valid(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
                true
            }
        }

        #[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
        struct TestTaskNoop;

        impl fmt::Display for TestTaskNoop {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "TestTaskNoop")
            }
        }

        impl Task<TestEngine> for TestTaskNoop {
            fn weight(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> f32 {
                1.
            }

            fn is_valid(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
                true
            }

            fn execute(
                &self,
                _state: StateDiffRefMut<TestEngine>,
                _agent: AgentId,
            ) -> Option<Box<dyn Task<TestEngine>>> {
                None
            }

            fn display_action(&self) -> <value_tests::TestEngine as Domain>::DisplayAction {
                ()
            }

            fn box_eq(&self, other: &Box<dyn Task<TestEngine>>) -> bool {
                other
                    .downcast_ref::<Self>()
                    .map(|other| self.eq(other))
                    .unwrap_or_default()
            }

            fn box_clone(&self) -> Box<dyn Task<TestEngine>> {
                Box::new(self.clone())
            }

            fn box_hash(&self, mut state: &mut dyn Hasher) {
                self.hash(&mut state)
            }
        }

        #[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
        struct TestTaskNegative;

        impl fmt::Display for TestTaskNegative {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "TestTaskNegative")
            }
        }

        impl Task<TestEngine> for TestTaskNegative {
            fn weight(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> f32 {
                1.
            }

            fn is_valid(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
                true
            }

            fn execute(
                &self,
                mut state: StateDiffRefMut<TestEngine>,
                _agent: AgentId,
            ) -> Option<Box<dyn Task<TestEngine>>> {
                state.add_value(-1);
                None
            }

            fn display_action(&self) -> <value_tests::TestEngine as Domain>::DisplayAction {
                ()
            }

            fn box_eq(&self, other: &Box<dyn Task<TestEngine>>) -> bool {
                other
                    .downcast_ref::<Self>()
                    .map(|other| self.eq(other))
                    .unwrap_or_default()
            }

            fn box_clone(&self) -> Box<dyn Task<TestEngine>> {
                Box::new(self.clone())
            }

            fn box_hash(&self, mut state: &mut dyn Hasher) {
                self.hash(&mut state)
            }
        }

        #[test]
        fn negative() {
            for depth in (5..=20).step_by(5) {
                let mut noop = 0;
                let mut neg = 0;

                for _ in 0..20 {
                    let config = MCTSConfiguration {
                        visits: 1.5f32.powi(depth as i32) as u32 / 10 + 100,
                        depth,
                        exploration: 1.414,
                        discount: 0.95,
                        seed: None
                    };
                    let agent = AgentId(0);

                    let state = State {
                        value: Default::default(),
                    };
                    let mut mcts = MCTS::<TestEngine>::new(
                        state,
                        agent,
                        &BTreeMap::new(),
                        config,
                    );

                    let task = mcts.run();
                    if task.downcast_ref::<TestTaskNoop>().is_some() {
                        noop += 1;
                    } else {
                        neg += 1;
                    }
                }

                assert!((noop as f32 / (noop + neg) as f32) >= 0.75);
            }
        }
    }
}
