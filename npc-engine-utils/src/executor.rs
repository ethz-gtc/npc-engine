use std::hash::Hash;
use npc_engine_common::{Domain, AgentId, ActiveTask, ActiveTasks, StateDiffRef, StateDiffRefMut, MCTS, MCTSConfiguration, IdleTask, StateValueEstimator, DefaultPolicyEstimator};

/// A domain for which we can run a generic executor
pub trait ExecutableDomain: Domain {
    /// Applies a diff to a mutable state
    fn apply_diff(diff: Self::Diff, state: &mut Self::State);
}
impl<
	S: std::fmt::Debug + Sized + Clone + Hash + Eq,
	DA: std::fmt::Debug + Default,
	D: Domain<State = S, Diff = Option<S>, DisplayAction = DA>,
> ExecutableDomain for D {
	fn apply_diff(diff: Self::Diff, state: &mut Self::State) {
		if let Some(diff) = diff {
			*state = diff;
		}
	}
}

/// User-defined properties for the executor, consisting of a set of
/// helper functions and possibly an associated state.
pub trait ExecutorState<D: ExecutableDomain> {
	/// Creates the state value estimator
	fn create_state_value_estimator(&self) -> Box<dyn StateValueEstimator<D>> {
		Box::new(DefaultPolicyEstimator {})
	}
	/// Creates the initial world state
	fn create_initial_state(&self) -> D::State;
	/// Fills the initial queue of tasks
	fn init_task_queue(&self) -> ActiveTasks<D>;
	/// Returns whether an agent should be kept in a given state (to remove dead agents)
	fn keep_agent(&self, tick: u64, state: &D::State, agent: AgentId) -> bool;
	/// Method called after action execution, to perform tasks such as visual updates and checking for newly-created agents
	fn post_action_execute_hook(&mut self, _state: &D::State, _diff: &D::Diff, _active_task: &ActiveTask<D>, _queue: &mut ActiveTasks<D>) {}
	/// Method called after MCTS has run, to perform tasks such as printing the search tree
	fn post_mcts_run_hook(&mut self, _mcts: &MCTS<D>, _last_active_task: &ActiveTask<D>) {}
}

/// A single-threaded generic executor
pub struct SimpleExecutor<'a, D, S>
	where
		D: ExecutableDomain,
		D::State: Clone,
		S: ExecutorState<D>
{
	/// The attached MCTS configuration
	pub mcts_config: MCTSConfiguration,
	/// The state of this executor
	pub executor_state: &'a mut S,
	/// The current state of the world
	pub state: D::State,
	/// The current queue of tasks
	pub task_queue: ActiveTasks<D>,
}
impl<'a, D, S> SimpleExecutor<'a, D, S>
	where
		D: ExecutableDomain,
		D::State: Clone,
		S: ExecutorState<D>
{
	/// Creates a new executor, initializes state and task queue from the CB trait
	pub fn new(mcts_config: MCTSConfiguration, executor_state: &'a mut S) -> Self {
		Self {
			mcts_config,
			state: executor_state.create_initial_state(),
			task_queue: executor_state.init_task_queue(),
			executor_state
		}
	}

	/// Executes one task, returns whether there are still tasks in the queue
	pub fn step(&mut self) -> bool {
		if self.task_queue.is_empty() {
			return false;
		}

		// Pop first task that is completed
		let active_task = self.task_queue.iter().next().unwrap().clone();
		self.task_queue.remove(&active_task);
		let active_agent = active_task.agent;
		let tick = active_task.end;

		// Should we continue considering that agent?
		if !self.executor_state.keep_agent(tick, &self.state, active_agent) {
			return true;
		}

		// Show state if in debug mode
		let mut diff = D::Diff::default();
		let state_diff = StateDiffRef::<D>::new(&self.state, &diff);
		if log::log_enabled!(log::Level::Info) {
			let highlight_style = ansi_term::Style::new().bold().fg(ansi_term::Colour::Green);
			let time_text = format!("T{}", tick);
			let task_name = format!("{:?}", active_task.task);
			let agent_id_text = format!("A{}", active_agent.0);
			log::info!("\n{}, State:\n{}\n{} task to be executed: {}", highlight_style.paint(&time_text), D::get_state_description(state_diff), highlight_style.paint(&agent_id_text), highlight_style.paint(&task_name));
		}

		// Execute task
		let is_task_valid = active_task.task.is_valid(tick, state_diff, active_agent);
		let new_task = if is_task_valid {
			log::info!("Valid task, executing...");
			let state_diff_mut = StateDiffRefMut::new(&self.state, &mut diff);
			let new_task = active_task.task.execute(tick, state_diff_mut, active_agent);
			self.executor_state.post_action_execute_hook(&self.state, &diff, &active_task, &mut self.task_queue);
			D::apply_diff(diff, &mut self.state);
			new_task
		} else {
			log::info!("Invalid task!");
			None
		};

		// If no next task, plan and get the task for this agent
		let new_task = new_task.unwrap_or_else(|| {
			log::info!("No subsequent task, planning!");
			let mut mcts = MCTS::<D>::new_with_tasks(
				self.state.clone(),
				active_agent,
				tick,
				self.task_queue.clone(),
				self.mcts_config.clone(),
				self.executor_state.create_state_value_estimator()
			);
			let new_task = mcts.run().unwrap_or_else(|| Box::new(IdleTask));
			self.executor_state.post_mcts_run_hook(&mcts, &active_task);
			new_task
		});

		// Add new task to queue
		let diff = D::Diff::default();
		let state_diff = StateDiffRef::new(&self.state, &diff);
		if log::log_enabled!(log::Level::Info) {
			let new_active_task = ActiveTask::new(active_agent, new_task.clone(), tick, state_diff);
			log::info!("Queuing new task for A{} until T{}: {:?}", active_agent.0, new_active_task.end, new_task);
		}
		let new_active_task = ActiveTask::new(active_agent, new_task, tick, state_diff);
		self.task_queue.insert(new_active_task);

		true
	}
}

/// Creates and runs a single-threaded executor, initializes state and task queue from the CB trait
pub fn run_simple_executor<D, S>(mcts_config: &MCTSConfiguration, executor_state: &mut S)
	where
		D: ExecutableDomain,
		D::State: Clone,
		S: ExecutorState<D>
{
	// Initialize the state and the agent queue
	let mut executor = SimpleExecutor::<D, S>::new(
		mcts_config.clone(),
		executor_state
	);
	loop {
		if !executor.step() {
			break;
		}
	}
}
