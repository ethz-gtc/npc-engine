use std::{collections::BTreeSet, marker::PhantomData};
use npc_engine_common::{Domain, AgentId, ActiveTask, StateDiffRef, StateDiffRefMut, MCTS, MCTSConfiguration, IdleTask};

type TaskQueue<D> = BTreeSet<ActiveTask<D>>;

/// A domain for which we can run a generic executor
pub trait ExecutableDomain: Domain {
    /// Applies a diff to a mutable state
    fn apply_diff(diff: &Self::Diff, state: &mut Self::State);
}

/// User-defined functions for the executor
pub trait ExecutorCallbacks<D: ExecutableDomain> {
	fn create_initial_state() -> D::State;
	fn init_queue() -> TaskQueue<D>;
	fn keep_agent(state: &D::State, agent: AgentId) -> bool;
	fn post_action_execute_hook(_state: &D::State, _diff: &D::Diff, _queue: &mut TaskQueue<D>) {}
	fn post_mcts_run_hook(_mcts: &MCTS<D>, _last_active_task: &ActiveTask<D>) {}
}


pub struct SimpleExecutor<D, CB>
	where
		D: ExecutableDomain,
		D::State: Clone,
		CB: ExecutorCallbacks<D>
{
	pub config: MCTSConfiguration,
	pub state: D::State,
	pub queue: TaskQueue<D>,
	phantom: PhantomData<CB>,
}
impl<D, CB> SimpleExecutor<D, CB>
	where
		D: ExecutableDomain,
		D::State: Clone,
		CB: ExecutorCallbacks<D>
{
	pub fn new(config: MCTSConfiguration) -> Self {
		Self {
			config,
			state: CB::create_initial_state(),
			queue: CB::init_queue(),
			phantom: PhantomData
		}
	}
	pub fn step(&mut self) -> bool {
		if self.queue.is_empty() {
			return false;
		}

		// Pop first task that is completed
		let active_task = self.queue.iter().next().unwrap().clone();
		self.queue.remove(&active_task);
		let active_agent = active_task.agent;
		let tick = active_task.end;

		// Should we continue considering that agent?
		if !CB::keep_agent(&self.state, active_agent) {
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
			CB::post_action_execute_hook(&self.state, &diff, &mut self.queue);
			D::apply_diff(&diff, &mut self.state);
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
				self.queue.clone(),
				self.config.clone()
			);
			let new_task = mcts.run().unwrap_or_else(|| Box::new(IdleTask));
			CB::post_mcts_run_hook(&mcts, &active_task);
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
		self.queue.insert(new_active_task);

		true
	}
}

/// A simple executor
pub fn run_simple_executor<D, CB>(config: &MCTSConfiguration)
	where
		D: ExecutableDomain,
		D::State: Clone,
		CB: ExecutorCallbacks<D>
{
	// Initialize the state and the agent queue
	let mut executor = SimpleExecutor::<D, CB>::new(config.clone());
	loop {
		if !executor.step() {
			break;
		}
	}
}
