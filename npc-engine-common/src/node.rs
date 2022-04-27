use std::{sync::{Arc, Weak}, collections::{BTreeMap, BTreeSet}, fmt, mem, hash::{Hash, Hasher}};

use crate::{Domain, AgentId, Task, StateDiffRef, AgentValue, active_task::{ActiveTask, ActiveTasks}, get_task_for_agent};

/// Strong atomic reference counted node
pub type Node<D> = Arc<NodeInner<D>>;

/// Weak atomic reference counted node
pub type WeakNode<D> = Weak<NodeInner<D>>;

pub struct NodeInner<D: Domain> {
    pub(crate) diff: D::Diff,
    pub(crate) active_agent: AgentId,
    pub(crate) tick: u64,
    pub(crate) tasks: ActiveTasks<D>,
    current_values: BTreeMap<AgentId, AgentValue>, // pre-computed current values
}

impl<D: Domain> fmt::Debug for NodeInner<D> {
    fn fmt(&self, f: &'_ mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("NodeInner")
            .field("diff", &self.diff)
            .field("agent", &self.active_agent)
            .field("tick", &self.tick)
            .field("tasks", &self.tasks)
            .field("current_values", &self.current_values)
            .finish()
    }
}

impl<D: Domain> NodeInner<D> {
    /// Create a new node, check for visible agents, and re-assign current tasks to the matching ones.
    /// Return None if no active agent is not visible, and Some(node) otherwise.
    pub fn new(
        initial_state: &D::State,
        start_tick: u64,
        diff: D::Diff,
        active_agent: AgentId,
        tick: u64,
        tasks: BTreeSet<ActiveTask<D>>,
    ) -> Self {
        let state_diff = StateDiffRef::new(initial_state, &diff);
        // Get list of agents we consider in planning
        let mut agents = tasks
            .iter()
            .map(|task| task.agent)
            .collect();
        D::update_visible_agents(
            start_tick,
            tick,
            state_diff,
            active_agent,
            &mut agents
        );

        // Assign idle tasks to agents without a task
        let (tasks, current_values): (ActiveTasks<D>, _) = agents
            .into_iter()
            .map(|agent|
                get_task_for_agent(&tasks, agent)
                    .map_or_else(
                        || ActiveTask::new_idle(tick, agent, active_agent),
                        |task| task.clone()
                    )
            )
            // Set child current values
            .map(|task| {
                let agent = task.agent;
                (
                    task,
                    (
                        agent,
                        D::get_current_value(
                            tick,
                            state_diff,
                            agent
                        )
                    )
                )
            })
            .unzip();

        NodeInner {
            active_agent,
            diff,
            tick,
            tasks,
            current_values
        }
    }

    /// Return agent who owns the node
    pub fn agent(&self) -> AgentId {
        self.active_agent
    }

    /// Return the tick
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Return all agents that are in considered by this node
    pub fn agents(&self) -> BTreeSet<AgentId> {
        self.tasks
            .iter()
            .map(|task| task.agent)
            .collect()
    }

    /// Return diff of current node.
    pub fn diff(&self) -> &D::Diff {
        &self.diff
    }

    /// Return the current value from an agent, panic if not present in the node
    pub fn current_value(&self, agent: AgentId) -> AgentValue {
        self.current_values.get(&agent).copied().unwrap_or_else(|| AgentValue::new(0.0).unwrap())
    }

    /// Return the current value from an agent, compute if not present in the node
    pub fn current_value_or_compute(&self, agent: AgentId, initial_state: &D::State) -> AgentValue {
        self.current_values
            .get(&agent)
            .copied()
            .unwrap_or_else(||
                D::get_current_value(
                    self.tick,
                    StateDiffRef::new(
                        initial_state,
                        &self.diff,
                    ),
                    agent,
                )
            )
    }

    /// Return all current values
    pub fn current_values(&self) -> &BTreeMap<AgentId, AgentValue> {
        &self.current_values
    }

    // Returns the size in bytes
    pub fn size(&self, task_size: fn(&dyn Task<D>) -> usize) -> usize {
        let mut size = 0;

        size += mem::size_of::<Self>();
        size += self.current_values.len() * mem::size_of::<(AgentId, f32)>();

        for task in &self.tasks {
            size += task.size(task_size);
        }

        size
    }
}

impl<D: Domain> Hash for NodeInner<D> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.active_agent.hash(hasher);
        self.diff.hash(hasher);
        self.tasks.hash(hasher);
        self.tick.hash(hasher);
    }
}

impl<D: Domain> PartialEq for NodeInner<D> {
    fn eq(&self, other: &Self) -> bool {
        self.active_agent.eq(&other.active_agent) && self.diff.eq(&other.diff) && self.tasks.eq(&other.tasks) && self.tick.eq(&other.tick)
    }
}

impl<D: Domain> Eq for NodeInner<D> {}
