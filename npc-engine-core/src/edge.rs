/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::{
    collections::BTreeSet,
    fmt, mem,
    ops::Range,
    sync::{Arc, Mutex},
};

use crate::{AgentId, AgentValue, Context, Domain, Node, SeededHashMap, Task, WeakNode};

use rand::distributions::WeightedIndex;

/// The tasks left to expand in a given node.
///
/// None if all tasks are expanded.
type UnexpandedTasks<D> = Option<(WeightedIndex<f32>, Vec<Box<dyn Task<D>>>)>;

/// The outgoing edges from a node, possibly partially expanded.
pub struct Edges<D: Domain> {
    pub(crate) unexpanded_tasks: UnexpandedTasks<D>,
    pub(crate) expanded_tasks: SeededHashMap<Box<dyn Task<D>>, Edge<D>>,
}
impl<D: Domain> fmt::Debug for Edges<D> {
    fn fmt(&self, f: &'_ mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Edges")
            .field("unexpanded_tasks", &self.unexpanded_tasks)
            .field("expanded_tasks", &self.expanded_tasks)
            .finish()
    }
}

impl<'a, D: Domain> IntoIterator for &'a Edges<D> {
    type Item = (&'a Box<dyn Task<D>>, &'a Edge<D>);
    type IntoIter = std::collections::hash_map::Iter<'a, Box<dyn Task<D>>, Edge<D>>;

    fn into_iter(self) -> Self::IntoIter {
        self.expanded_tasks.iter()
    }
}

impl<D: Domain> Edges<D> {
    /// Creates new edges, with optionally a forced task that will be the sole edge.
    pub fn new(
        node: &Node<D>,
        initial_state: &D::State,
        next_task: Option<Box<dyn Task<D>>>,
    ) -> Self {
        let ctx =
            Context::with_state_and_diff(node.tick, initial_state, &node.diff, node.active_agent);
        let unexpanded_tasks = match next_task {
            Some(task) if task.is_valid(ctx) => {
                let weights = WeightedIndex::new([1.].iter().map(Clone::clone)).unwrap();

                // Set existing child weights, only option
                Some((weights, vec![task.clone()]))
            }
            _ => {
                // Get possible tasks
                let tasks = D::get_tasks(ctx);
                if tasks.is_empty() {
                    // no task, return empty edges
                    return Edges {
                        unexpanded_tasks: None,
                        expanded_tasks: Default::default(),
                    };
                }

                // Safety-check that all tasks are valid
                for task in &tasks {
                    debug_assert!(task.is_valid(ctx));
                }

                // Get the weight for each task
                let weights =
                    WeightedIndex::new(tasks.iter().map(|task| task.weight(ctx))).unwrap();

                Some((weights, tasks))
            }
        };

        Edges {
            unexpanded_tasks,
            expanded_tasks: Default::default(),
        }
    }

    /// Returns the sum of all visits to the edges of this nodes.
    pub fn child_visits(&self) -> usize {
        self.expanded_tasks
            .values()
            .map(|edge| edge.lock().unwrap().visits)
            .sum()
    }

    /// Finds the best task with the given `exploration` factor and normalization `range`.
    pub fn best_task(
        &self,
        agent: AgentId,
        exploration: f32,
        range: Range<AgentValue>,
    ) -> Option<Box<dyn Task<D>>> {
        let visits = self.child_visits();
        self.expanded_tasks
            .iter()
            .max_by(|(_, a), (_, b)| {
                let a = a.lock().unwrap();
                let b = b.lock().unwrap();
                a.uct(agent, visits, exploration, range.clone())
                    .partial_cmp(&b.uct(agent, visits, exploration, range.clone()))
                    .unwrap()
            })
            .map(|(k, _)| k.clone())
    }

    /// Returns the weighted average q value of all child edges.
    ///
    /// The `fallback` value is used for self-referential edges.
    pub fn q_value(&self, fallback: (usize, f32), agent: AgentId) -> Option<f32> {
        self.expanded_tasks
            .values()
            .map(|edge| {
                edge.try_lock()
                    .map(|edge| {
                        (
                            edge.visits,
                            edge.q_values.get(&agent).copied().unwrap_or_default(),
                        )
                    })
                    .unwrap_or(fallback)
            })
            .fold(None, |acc, (visits, value)| match acc {
                Some((sum, count)) => Some((sum + visits as f32 * value, count + visits)),
                None => Some((visits as f32 * value, visits)),
            })
            .map(|(sum, count)| sum / count as f32)
    }

    /// Returns the number of already-expanded edges.
    pub fn expanded_count(&self) -> usize {
        self.expanded_tasks.len()
    }

    /// Returns the number of not-yet-expanded edges.
    pub fn unexpanded_count(&self) -> usize {
        self.unexpanded_tasks
            .as_ref()
            .map_or(0, |(_, tasks)| tasks.len())
    }

    /// Returns how many edges there are, the sum of the expanded and not-yet expanded counts.
    pub fn branching_factor(&self) -> usize {
        self.expanded_count() + self.unexpanded_count()
    }

    /// Returns the expanded edge associated to a task, None if it does not exist.
    #[allow(clippy::borrowed_box)]
    pub fn get_edge(&self, task: &Box<dyn Task<D>>) -> Option<Edge<D>> {
        self.expanded_tasks.get(task).cloned()
    }

    /// The memory footprint of this struct.
    pub fn size(&self, task_size: fn(&dyn Task<D>) -> usize) -> usize {
        let mut size = 0;

        size += mem::size_of::<Self>();

        if let Some((_, tasks)) = self.unexpanded_tasks.as_ref() {
            for task in tasks {
                size += task_size(&**task);
            }
        }

        for (task, edge) in &self.expanded_tasks {
            size += task_size(&**task);
            size += edge.lock().unwrap().size();
        }

        size
    }
}

/// Strong atomic reference counted edge.
pub type Edge<D> = Arc<Mutex<EdgeInner<D>>>;

/// The data associated with an edge.
pub struct EdgeInner<D: Domain> {
    pub(crate) parent: WeakNode<D>,
    pub(crate) child: WeakNode<D>,
    pub(crate) visits: usize,
    pub(crate) q_values: SeededHashMap<AgentId, f32>,
}

impl<D: Domain> fmt::Debug for EdgeInner<D> {
    fn fmt(&self, f: &'_ mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("EdgeInner")
            .field("parent", &self.parent)
            .field("child", &self.child)
            .field("visits", &self.visits)
            .field("q_values", &self.q_values)
            .finish()
    }
}

/// Creates a new edge between a parent and a child.
pub(crate) fn new_edge<D: Domain>(
    parent: &Node<D>,
    child: &Node<D>,
    agents: &BTreeSet<AgentId>,
) -> Edge<D> {
    Arc::new(Mutex::new(EdgeInner {
        parent: Node::downgrade(parent),
        child: Node::downgrade(child),
        visits: Default::default(),
        q_values: agents.iter().map(|agent| (*agent, 0.)).collect(),
    }))
}

impl<D: Domain> EdgeInner<D> {
    /// Calculates the current UCT value for the edge.
    pub fn uct(
        &self,
        parent_agent: AgentId,
        parent_child_visits: usize,
        exploration: f32,
        range: Range<AgentValue>,
    ) -> f32 {
        // If parent is not present, this node is being reused and the parent leaves the horizon. Score doesn't matter
        if let Some(q_value) = self.q_values.get(&parent_agent) {
            // Normalize the exploitation factor so it doesn't overshadow the exploration
            let exploitation_value =
                (q_value - *range.start) / (*(range.end - range.start)).max(f32::EPSILON);
            let exploration_value =
                ((parent_child_visits as f32).ln() / (self.visits as f32).max(f32::EPSILON)).sqrt();
            exploitation_value + exploration * exploration_value
        } else {
            0.
        }
    }

    /// Returns the number of visits to this edge
    pub fn visits(&self) -> usize {
        self.visits
    }

    /// Get the q-value of a given agent, 0 if not present
    pub fn q_value(&self, agent: AgentId) -> f32 {
        self.q_values.get(&agent).copied().unwrap_or(0.)
    }

    /// Returns the linked child node.
    pub fn child(&self) -> Node<D> {
        self.child.upgrade().unwrap()
    }

    /// Returns the linked parent node.
    pub fn parent(&self) -> Node<D> {
        self.parent.upgrade().unwrap()
    }

    /// The memory footprint of this struct.
    pub fn size(&self) -> usize {
        let mut size = 0;

        size += mem::size_of::<Self>();
        size += self.q_values.len() * mem::size_of::<(AgentId, f32)>();

        size
    }
}
