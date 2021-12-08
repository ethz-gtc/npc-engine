use std::{sync::{Arc, Weak}, collections::BTreeMap, fmt, mem, hash::{Hash, Hasher}};

use crate::{Domain, AgentId, Task, SnapshotDiffRef};

/// Strong atomic reference counted node
pub type Node<D> = Arc<NodeInner<D>>;

/// Weak atomic reference counted node
pub type WeakNode<D> = Weak<NodeInner<D>>;

// FIXME: unpub
pub struct NodeInner<D: Domain> {
    pub diff: D::Diff,
    pub agent: AgentId,
    pub tasks: BTreeMap<AgentId, Box<dyn Task<D>>>,
    pub current_values: BTreeMap<AgentId, f32>,
}

impl<D: Domain> fmt::Debug for NodeInner<D> {
    fn fmt(&self, f: &'_ mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("NodeInner")
            .field("diff", &self.diff)
            .field("agent", &self.agent)
            .field("tasks", &"...")
            .field("current_values", &self.current_values)
            .finish()
    }
}

impl<D: Domain> NodeInner<D> {
    pub fn new(
        snapshot: &D::Snapshot,
        diff: D::Diff,
        agent: AgentId,
        mut tasks: BTreeMap<AgentId, Box<dyn Task<D>>>,
    ) -> Self {
        // Check validity of task for agent
        if let Some(task) = tasks.get(&agent) {
            if !task.is_valid(SnapshotDiffRef::new(snapshot, &diff), agent) {
                tasks.remove(&agent);
            }
        }

        // Get observable agents
        let agents = D::get_visible_agents(
            SnapshotDiffRef::new(snapshot, &diff),
            agent
        );

        // Set child current values
        let current_values = agents
            .iter()
            .map(|agent| {
                (
                    *agent,
                    D::get_current_value(SnapshotDiffRef::new(snapshot, &diff), *agent),
                )
            })
            .collect();

        NodeInner {
            agent,
            diff,
            tasks,
            current_values,
        }
    }

    /// Returns agent who owns the node.
    pub fn agent(&self) -> AgentId {
        self.agent
    }

    /// Returns diff of current node.
    pub fn diff(&self) -> &D::Diff {
        &self.diff
    }

    // Returns the size in bytes
    pub fn size(&self, task_size: fn(&dyn Task<D>) -> usize) -> usize {
        let mut size = 0;

        size += mem::size_of::<Self>();
        size += self.current_values.len() * mem::size_of::<(AgentId, f32)>();

        for (_, task) in &self.tasks {
            size += task_size(&**task);
        }

        size
    }
}

impl<D: Domain> Hash for NodeInner<D> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.agent.hash(hasher);
        self.diff.hash(hasher);
        self.tasks.hash(hasher);
    }
}

impl<D: Domain> PartialEq for NodeInner<D> {
    fn eq(&self, other: &Self) -> bool {
        self.agent.eq(&other.agent) && self.diff.eq(&other.diff) && self.tasks.eq(&other.tasks)
    }
}

impl<D: Domain> Eq for NodeInner<D> {}
