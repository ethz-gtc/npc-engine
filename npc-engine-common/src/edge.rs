use std::{ops::Range, mem, sync::Arc, cell::RefCell, fmt, collections::BTreeSet};

use crate::{Domain, AgentId, SeededHashMap, Task, StateDiffRef, Node, WeakNode, AgentValue};

use rand::distributions::WeightedIndex;

pub type UnexpandedTasks<D> = Option<(WeightedIndex<f32>, Vec<Box<dyn Task<D>>>)>;

pub struct Edges<D: Domain> {
    pub unexpanded_tasks: UnexpandedTasks<D>,
    pub expanded_tasks: SeededHashMap<Box<dyn Task<D>>, Edge<D>>,
}

impl<'a, D: Domain> IntoIterator for &'a Edges<D> {
    type Item = (&'a Box<dyn Task<D>>, &'a Edge<D>);
    type IntoIter = std::collections::hash_map::Iter<'a, Box<dyn Task<D>>, Edge<D>>;

    fn into_iter(self) -> Self::IntoIter {
        self.expanded_tasks.iter()
    }
}

impl<D: Domain> Edges<D> {
    /// Create new edges, with optionally a task to continue
    pub fn new(node: &Node<D>, initial_state: &D::State, next_task: Option<Box<dyn Task<D>>>) -> Self {

        let unexpanded_tasks = match next_task {
            Some(task) if task.is_valid(node.tick, StateDiffRef::new(initial_state, &node.diff), node.active_agent) => {
                let weights = WeightedIndex::new((&[1.]).iter().map(Clone::clone)).unwrap();

                // Set existing child weights, only option
                Some((weights, vec![task.clone()]))
            }
            _ => {
                // Get possible tasks
                let tasks = D::get_tasks(
                    node.tick,
                    StateDiffRef::new(initial_state, &node.diff),
                    node.active_agent
                );
                if tasks.is_empty() {
                    // no task, return empty edges
                    return Edges {
                        unexpanded_tasks: None,
                        expanded_tasks: Default::default()
                    }
                }

                // Safety-check that all tasks are valid
                let state_diff = StateDiffRef::new(initial_state, &node.diff);
                for task in &tasks {
                    debug_assert!(task.is_valid(node.tick, state_diff, node.active_agent));
                }

                // Get the weight for each task
                let weights =
                    WeightedIndex::new(tasks.iter().map(|task| {
                        task.weight(node.tick, state_diff, node.active_agent)
                    }))
                    .unwrap();

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
            .map(|edge| edge.borrow().visits)
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
                let a = a.borrow();
                let b = b.borrow();
                a.uct(agent, visits, exploration, range.clone())
                    .partial_cmp(&b.uct(agent, visits, exploration, range.clone()))
                    .unwrap()
            })
            .map(|(k, _)| k.clone())
    }

    /// Returns the weighted average q value of all child edges.
    /// `fallback` value is for self-referential edges.
    pub fn value(&self, fallback: (usize, f32), agent: AgentId) -> Option<f32> {
        self.expanded_tasks
            .values()
            .map(|edge| {
                edge.try_borrow()
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

    pub fn branching_factor(&self) -> usize {
        self.expanded_tasks.len() +
        self.unexpanded_tasks.as_ref().map_or(0,
            |(_, tasks)| tasks.len()
        )
    }

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
            size += edge.borrow().size();
        }

        size
    }
}

pub type Edge<D> = Arc<RefCell<EdgeInner<D>>>;

pub struct EdgeInner<D: Domain> {
    pub parent: WeakNode<D>,
    pub child: WeakNode<D>,
    pub visits: usize,
    pub q_values: SeededHashMap<AgentId, f32>,
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

pub fn new_edge<D: Domain>(parent: &Node<D>, child: &Node<D>, agents: &BTreeSet<AgentId>) -> Edge<D> {
    Arc::new(RefCell::new(EdgeInner {
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
            let exploitation_value = (q_value - *range.start) / (*(range.end - range.start)).max(f32::EPSILON);
            let exploration_value = ((parent_child_visits as f32).ln() / (self.visits as f32).max(f32::EPSILON)).sqrt();
            exploitation_value + exploration * exploration_value
        } else {
            0.
        }
    }

    pub fn size(&self) -> usize {
        let mut size = 0;

        size += mem::size_of::<Self>();
        size += self.q_values.len() * mem::size_of::<(AgentId, f32)>();

        size
    }
}