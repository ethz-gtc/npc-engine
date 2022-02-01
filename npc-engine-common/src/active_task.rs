use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::{fmt, mem};
use std::hash::{Hash, Hasher};
use crate::{AgentId, Domain, Task, StateDiffRef, IdleTask};

pub struct ActiveTask<D: Domain> {
    pub end: u64,
	pub agent: AgentId,
    pub task: Box<dyn Task<D>>,
}
pub type ActiveTasks<D> = BTreeSet<ActiveTask<D>>;

impl<D: Domain> ActiveTask<D> {
	pub fn new(agent: AgentId, task: Box<dyn Task<D>>, tick: u64, state_diff: StateDiffRef<D>) -> ActiveTask<D> {
		let end = tick + task.duration(tick, state_diff, agent);//.get();
		ActiveTask {
			end,
			agent,
			task,
		}
	}
    /// Create a new idle task for agent at a given tick, make sure that it will
    /// execute in the future considering that we are currently processing active_agent.
    pub fn new_idle(tick: u64, agent: AgentId, active_agent: AgentId) -> ActiveTask<D> {
        ActiveTask {
            // Make sure the idle tasks of added agents will not be
            // executed before the active agent
            end: if agent < active_agent { tick + 1 } else { tick },
            agent,
            task: Box::new(IdleTask)
        }
    }
	pub fn size(&self, task_size: fn(&dyn Task<D>) -> usize) -> usize {
		let mut size = 0;
		size += mem::size_of::<Self>();
		size += task_size(&*self.task);
		size
	}
}
pub fn contains_agent<D: Domain>(set: &BTreeSet<ActiveTask<D>>, agent: AgentId) -> bool {
	for task in set {
		if task.agent == agent {
			return true;
		}
	}
	false
}
pub fn get_task_for_agent<D: Domain>(set: &BTreeSet<ActiveTask<D>>, agent: AgentId) -> Option<&ActiveTask<D>> {
    for task in set {
		if task.agent == agent {
			return Some(task)
		}
	}
	None
}

impl<D: Domain> fmt::Debug for ActiveTask<D> {
    fn fmt(&self, f: &'_ mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ActiveTask")
            .field("end", &self.end)
            .field("agent", &self.agent)
            .field("task", &self.task)
            .finish()
    }
}

impl<D: Domain> Hash for ActiveTask<D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.end.hash(state);
        self.agent.hash(state);
		self.task.hash(state);
    }
}

impl<D: Domain> Clone for ActiveTask<D> {
    fn clone(&self) -> Self {
        ActiveTask {
            end: self.end,
            agent: self.agent,
            task: self.task.box_clone(),
        }
    }
}

impl<D: Domain> Ord for ActiveTask<D> {
	/// Active tasks are ordered first by time of ending, then by agent id
    fn cmp(&self, other: &Self) -> Ordering {
        self.end.cmp(&other.end).then(self.agent.cmp(&other.agent))
    }
}

impl<D: Domain> PartialOrd for ActiveTask<D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<D: Domain> Eq for ActiveTask<D> {}

impl<D: Domain> PartialEq for ActiveTask<D> {
    fn eq(&self, other: &Self) -> bool {
        self.end == other.end &&
        self.agent == other.agent &&
        self.task.box_eq(&other.task)
    }
}