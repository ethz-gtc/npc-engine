use std::{sync::{Arc, Weak}, collections::{BTreeMap, BTreeSet}, fmt, mem, hash::{Hash, Hasher}};

use crate::{Domain, AgentId, Task, StateDiffRef, AgentValue, active_task::{ActiveTask, contains_agent, ActiveTasks}};

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
            .field("tasks", &self.tasks)
            .field("current_values", &self.current_values)
            .finish()
    }
}

impl<D: Domain> NodeInner<D> {
    /// Create a new node, extract agent list from active tasks
    pub fn new(
        initial_state: &D::State,
        diff: D::Diff,
        active_agent: AgentId,
        tick: u64,
        tasks: BTreeSet<ActiveTask<D>>,
    ) -> Self {
        debug_assert!(contains_agent(&tasks, active_agent));

        // Set child current values
        let current_values = tasks
            .iter()
            .map(|task| {
                (
                    task.agent,
                    D::get_current_value(StateDiffRef::new(initial_state, &diff), task.agent),
                )
            })
            .collect();

        NodeInner {
            active_agent,
            diff,
            tick,
            tasks,
            current_values
        }
    }

    /// Returns agent who owns the node.
    pub fn agent(&self) -> AgentId {
        self.active_agent
    }

    /// Return all agents that are in considered by this node
    pub fn agents(&self) -> BTreeSet<AgentId> {
        self.tasks
            .iter()
            .map(|task| task.agent)
            .collect()
    }

    /// Returns diff of current node.
    pub fn diff(&self) -> &D::Diff {
        &self.diff
    }

    /// Return the current value from an agent, panic if not present in the node
    pub fn current_value(&self, agent: AgentId) -> AgentValue {
        *self.current_values.get(&agent).unwrap()
    }

    /// Return the current value from an agent, compute if not present in the node
    pub fn current_value_or_compute(&self, agent: AgentId, initial_state: &D::State) -> AgentValue {
        self.current_values
            .get(&agent)
            .copied()
            .unwrap_or_else(||
                D::get_current_value(
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
    }
}

impl<D: Domain> PartialEq for NodeInner<D> {
    fn eq(&self, other: &Self) -> bool {
        self.active_agent.eq(&other.active_agent) && self.diff.eq(&other.diff) && self.tasks.eq(&other.tasks)
    }
}

impl<D: Domain> Eq for NodeInner<D> {}
