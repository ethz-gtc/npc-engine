/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use ansi_term::Style;
use npc_engine_core::{
    ActiveTask, ActiveTasks, AgentId, DefaultPolicyEstimator, Domain, DomainWithPlanningTask,
    EarlyStopCondition, IdleTask, MCTSConfiguration, PlanningTask, StateDiffRef, StateDiffRefMut,
    StateValueEstimator, Task, MCTS,
};
use std::{
    collections::HashMap,
    hash::Hash,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use crate::GlobalDomain;

fn highlight_style() -> Style {
    ansi_term::Style::new().bold().fg(ansi_term::Colour::Green)
}
fn highlight_tick(tick: u64) -> String {
    let tick_text = format!("T{}", tick);
    highlight_style().paint(&tick_text).to_string()
}

fn highlight_agent(agent_id: AgentId) -> String {
    let tick_text = format!("{}", agent_id);
    highlight_style().paint(&tick_text).to_string()
}

/// A domain for which we can run a generic executor.
pub trait ExecutableDomain: Domain {
    /// Applies a diff to a mutable state.
    fn apply_diff(diff: Self::Diff, state: &mut Self::State);
}
// automatic implementation for domains where diff is an option of state
impl<
        S: std::fmt::Debug + Sized + Clone + Hash + Eq,
        DA: std::fmt::Debug + Default,
        D: Domain<State = S, Diff = Option<S>, DisplayAction = DA>,
    > ExecutableDomain for D
{
    fn apply_diff(diff: Self::Diff, state: &mut Self::State) {
        if let Some(diff) = diff {
            *state = diff;
        }
    }
}

/// User-defined methods for the executor.
///
/// If you have no special need, you can just implement this trait on an empty struct:
/// ```
/// struct MyExecutorState;
/// impl ExecutorState<MyDomain> for MyExecutorState {}
/// ```
pub trait ExecutorState<D: Domain> {
    /// Creates the state value estimator (by default uses rollout-based simulation).
    fn create_state_value_estimator(&self) -> Box<dyn StateValueEstimator<D> + Send> {
        Box::new(DefaultPolicyEstimator {})
    }
    /// Method called after action execution, to perform tasks such as visual updates and checking for newly-created agents (by default does nothing).
    fn post_action_execute_hook(
        &mut self,
        _state: &D::State,
        _diff: &D::Diff,
        _active_task: &ActiveTask<D>,
        _queue: &mut ActiveTasks<D>,
    ) {
    }
    /// Method called after MCTS has run, to perform tasks such as printing the search tree (by default does nothing).
    fn post_mcts_run_hook(&mut self, _mcts: &MCTS<D>, _last_active_task: &ActiveTask<D>) {}
}

/// User-defined methods for the executor,
/// where world and planner states are similar.
///
/// This trait is used with the [SimpleExecutor].
/// You must implement at least [create_initial_state](Self::create_initial_state)
/// and [init_task_queue](Self::init_task_queue),
/// to create the initial state and build the initial list of tasks from it.
pub trait ExecutorStateLocal<D: Domain> {
    /// Creates the initial world state.
    fn create_initial_state(&self) -> D::State;
    /// Fills the initial queue of tasks.
    fn init_task_queue(&self, state: &D::State) -> ActiveTasks<D>;
    /// Returns whether an agent should be kept in a given state (to remove dead agents) (by default returns true).
    fn keep_agent(&self, _tick: u64, _state: &D::State, _agent: AgentId) -> bool {
        true
    }
}

/// User-defined methods for the executor,
/// where world and planner states are different.
///
/// This trait is typically used with the [ThreadedExecutor].
/// You must implement at least [create_initial_state](Self::create_initial_state)
/// and [init_task_queue](Self::init_task_queue),
/// to create the initial state and build the initial list of tasks from it.
pub trait ExecutorStateGlobal<D: GlobalDomain> {
    /// Creates the initial world state.
    fn create_initial_state(&self) -> D::GlobalState;
    /// Fills the initial queue of tasks.
    fn init_task_queue(&self, state: &D::GlobalState) -> ActiveTasks<D>;
    /// Returns whether an agent should be kept in a given state (to remove dead agents) (by default returns true).
    fn keep_agent(&self, _tick: u64, _state: &D::GlobalState, _agent: AgentId) -> bool {
        true
    }
    /// Returns whether execution should continue in given state (by default returns true).
    fn keep_execution(&self, _tick: u64, _queue: &ActiveTasks<D>, _state: &D::GlobalState) -> bool {
        true
    }
    /// Method called from [ThreadedExecutor::step] after all tasks have been executed at a given step (by default does nothing).
    fn post_step_hook(&mut self, _tick: u64, _state: &mut D::GlobalState) {}
}

/// The state of tasks undergoing execution.
///
/// This can be used directly to build your own executor,
/// or indirectly through [SimpleExecutor] or [ThreadedExecutor].
struct ExecutionQueue<D>
where
    D: Domain,
{
    /// The current queue of tasks
    task_queue: ActiveTasks<D>,
}
impl<D> ExecutionQueue<D>
where
    D: Domain,
{
    pub fn new(task_queue: ActiveTasks<D>) -> Self {
        Self { task_queue }
    }

    pub fn is_empty(&self) -> bool {
        self.task_queue.is_empty()
    }

    pub fn size(&self) -> usize {
        self.task_queue.len()
    }

    pub fn pop_first_task(&mut self) -> ActiveTask<D> {
        // TODO: use pop_first once it is stabilized, see: https://github.com/rust-lang/rust/issues/62924
        let active_task = self.task_queue.iter().next().unwrap().clone();
        self.task_queue.remove(&active_task);
        active_task
    }

    pub fn execute_task<S, C>(
        &mut self,
        active_task: &ActiveTask<D>,
        state: &D::State,
        executor_state: &mut S,
        mut new_agents_cb: C,
    ) -> (D::Diff, Option<Box<dyn Task<D>>>)
    where
        S: ExecutorState<D>,
        C: FnMut(&Vec<ActiveTask<D>>),
    {
        let active_agent = active_task.agent;
        let tick = active_task.end;

        // Show state if in debug mode
        let mut diff = D::Diff::default();
        let state_diff = StateDiffRef::<D>::new(state, &diff);
        if log::log_enabled!(log::Level::Info) {
            let highlight_style = highlight_style();
            let tick_text = format!("T{}", tick);
            let task_name = format!("{:?}", active_task.task);
            let agent_id_text = format!("A{}", active_agent.0);
            log::info!(
                "\n{}, State:\n{}\n{} task to be executed: {}",
                highlight_style.paint(&tick_text),
                D::get_state_description(state_diff),
                highlight_style.paint(&agent_id_text),
                highlight_style.paint(&task_name)
            );
        }

        // Execute task
        let is_task_valid = active_task.task.is_valid(tick, state_diff, active_agent);
        if is_task_valid {
            log::info!("Valid task, executing...");
            let state_diff_mut = StateDiffRefMut::new(state, &mut diff);
            let new_task = active_task.task.execute(tick, state_diff_mut, active_agent);
            // Get the new agents and create idle tasks for them,
            let mut new_agents_tasks = D::get_new_agents(StateDiffRef::new(state, &diff))
                .into_iter()
                .map(|new_agent| ActiveTask::new_idle(tick, new_agent, active_agent))
                .collect();
            // call-back the user with these tasks,
            new_agents_cb(&new_agents_tasks);
            // then add them to our queue,
            self.task_queue.extend(new_agents_tasks.drain(..));
            // then call the hook.
            executor_state.post_action_execute_hook(
                state,
                &diff,
                active_task,
                &mut self.task_queue,
            );
            (diff, new_task)
        } else {
            log::info!("Invalid task!");
            (diff, None)
        }
    }

    pub fn queue_task(
        &mut self,
        tick: u64,
        active_agent: AgentId,
        new_task: Box<dyn Task<D>>,
        state: &D::State,
    ) -> ActiveTask<D> {
        let diff = D::Diff::default();
        let state_diff = StateDiffRef::new(state, &diff);
        if log::log_enabled!(log::Level::Info) {
            let new_active_task = ActiveTask::new(active_agent, new_task.clone(), tick, state_diff);
            log::info!(
                "Queuing new task for {} until {}: {:?}",
                highlight_agent(active_agent),
                highlight_tick(new_active_task.end),
                new_task
            );
        }
        let new_active_task = ActiveTask::new(active_agent, new_task, tick, state_diff);
        self.task_queue.insert(new_active_task.clone());
        new_active_task
    }
}

/// A single-threaded generic executor.
///
/// It requies a domain `D` that knows how to apply a [Diff](Domain::Diff) to a [State](Domain::State),
/// and an executor state `S` that knows how to create the initial state
/// and build the initial list of tasks from it ([ExecutorStateLocal]).
/// The helper function [run_simple_executor] can create and run it for you.
pub struct SimpleExecutor<'a, D, S>
where
    D: ExecutableDomain,
    D::State: Clone,
    S: ExecutorState<D> + ExecutorStateLocal<D>,
{
    /// The attached MCTS configuration
    mcts_config: MCTSConfiguration,
    /// The state of this executor
    executor_state: &'a mut S,
    /// The current state of the world
    state: D::State,
    /// The current queue of tasks
    queue: ExecutionQueue<D>,
}
impl<'a, D, S> SimpleExecutor<'a, D, S>
where
    D: ExecutableDomain,
    D::State: Clone,
    S: ExecutorState<D> + ExecutorStateLocal<D>,
{
    /// Creates a new executor, initializes state and task queue from the S trait.
    pub fn new(mcts_config: MCTSConfiguration, executor_state: &'a mut S) -> Self {
        let state = executor_state.create_initial_state();
        let task_queue = executor_state.init_task_queue(&state);
        let queue = ExecutionQueue::new(task_queue);
        Self {
            mcts_config,
            state,
            queue,
            executor_state,
        }
    }

    /// Executes one task, returns whether there are still tasks in the queue.
    pub fn step(&mut self) -> bool {
        if self.queue.is_empty() {
            return false;
        }

        // Pop first task that is completed
        let active_task = self.queue.pop_first_task();
        let active_agent = active_task.agent;
        let tick = active_task.end;

        // Should we continue considering that agent?
        if !self
            .executor_state
            .keep_agent(tick, &self.state, active_agent)
        {
            return true;
        }

        // Execute the task and queue the new agents
        let (diff, new_task) =
            self.queue
                .execute_task(&active_task, &self.state, self.executor_state, |_| {});
        D::apply_diff(diff, &mut self.state);

        // If no next task, plan and get the task for this agent
        let new_task = new_task.unwrap_or_else(|| {
            log::info!("No subsequent task, planning!");
            let mut mcts = self.new_mcts(tick, active_agent);
            let new_task = mcts.run().unwrap_or_else(|| Box::new(IdleTask));
            self.executor_state.post_mcts_run_hook(&mcts, &active_task);
            new_task
        });

        // Add new task to queue
        self.queue
            .queue_task(tick, active_agent, new_task, &self.state);

        true
    }

    fn new_mcts(&self, tick: u64, active_agent: AgentId) -> MCTS<D> {
        MCTS::<D>::new_with_tasks(
            self.state.clone(),
            active_agent,
            tick,
            self.queue.task_queue.clone(),
            self.mcts_config.clone(),
            self.executor_state.create_state_value_estimator(),
            None,
        )
    }
}

/// Creates and runs a single-threaded executor, initializes state and task queue from the `S` trait.
pub fn run_simple_executor<D, S>(mcts_config: &MCTSConfiguration, executor_state: &mut S)
where
    D: ExecutableDomain,
    D::State: Clone,
    S: ExecutorState<D> + ExecutorStateLocal<D>,
{
    // Initialize the state and the agent queue
    let mut executor = SimpleExecutor::<D, S>::new(mcts_config.clone(), executor_state);
    loop {
        if !executor.step() {
            break;
        }
    }
}

/// A multi-threaded generic executor.
///
/// It maintains a [D::GlobalState](GlobalDomain::GlobalState) out of which a
/// [D::State](Domain::State) is [derived](GlobalDomain::derive_local_state) for planning.
/// This allows to simulate a large world with many agents, each of them planning on a small
/// subset of that world.
/// The helper function [run_threaded_executor] can create and run it for you.
pub struct ThreadedExecutor<'a, D, S>
where
    D: DomainWithPlanningTask + GlobalDomain,
    D::State: Clone + Send,
    D::Diff: Send + Sync,
    S: ExecutorState<D> + ExecutorStateGlobal<D>,
{
    /// The attached MCTS configuration
    mcts_config: MCTSConfiguration,
    /// The state of this executor
    executor_state: &'a mut S,
    /// The current state of the world
    state: D::GlobalState,
    /// The current queue of tasks
    queue: ExecutionQueue<D>,
    /// The last tasks which were executed
    /// Assuming that planning tasks all take the same time, we need to only
    /// keep one ActiveTask per AgentId, because after the first task choice,
    /// either it is planning or it is a forced task, which we do not put in the history.
    task_history: HashMap<AgentId, ActiveTask<D>>,
    /// Stores all planning threads
    threads: HashMap<AgentId, JoinHandle<MCTS<D>>>,
    /// The global tick of the simulation
    tick: Arc<AtomicU64>,
}
impl<'a, D, S> ThreadedExecutor<'a, D, S>
where
    D: DomainWithPlanningTask + GlobalDomain,
    D::State: Clone + Send,
    D::Diff: Send + Sync,
    S: ExecutorState<D> + ExecutorStateGlobal<D>,
{
    /// Creates a new executor, initializes state and task queue from the S trait.
    pub fn new(mcts_config: MCTSConfiguration, executor_state: &'a mut S) -> Self {
        let state = executor_state.create_initial_state();
        let task_queue = executor_state.init_task_queue(&state);
        let task_history = task_queue
            .iter()
            .map(|active_task| (active_task.agent, active_task.clone()))
            .collect();
        let queue = ExecutionQueue::new(task_queue);
        Self {
            mcts_config,
            state,
            queue,
            task_history,
            threads: Default::default(),
            tick: Arc::new(AtomicU64::new(0)),
            executor_state,
        }
    }

    fn new_mcts(&self, tick: u64, active_agent: AgentId) -> MCTS<D> {
        let planning_task_duration = self
            .mcts_config
            .planning_task_duration
            .expect("Planning task must have non-zero duration for threaded executor");
        let tick_atomic = self.tick.clone();
        let early_stop_condition: Option<Box<EarlyStopCondition>> = Some(Box::new(move || {
            tick_atomic.load(Ordering::Relaxed) >= tick + planning_task_duration.get() - 1
        }));
        MCTS::<D>::new_with_tasks(
            D::derive_local_state(&self.state, active_agent),
            active_agent,
            tick,
            self.queue.task_queue.clone(),
            self.mcts_config.clone(),
            self.executor_state.create_state_value_estimator(),
            early_stop_condition,
        )
    }

    /// Blocks on all planning threads which should have finished in the current tick and adds the
    /// resulting best tasks to the `active_tasks`.
    fn block_on_planning(&mut self, tick: u64) {
        // Iterate over all planning tasks that should have finished by now
        let active_tasks = self.queue.task_queue.clone();
        for active_task in active_tasks
            .iter()
            .filter(|task| task.end <= tick && task.task.downcast_ref::<PlanningTask>().is_some())
        {
            let active_agent = active_task.agent;
            debug_assert!(active_task.end == tick,
                "Processing an active planning task at tick {tick} but it should have been processed at tick {}.", active_task.end
            );

            // Try to get the planning thread of the current agent
            let thread = self.threads.remove(&active_agent);
            assert!(thread.is_some(),
                "There is no planning thread for {active_agent} even though there is an active_task for it."
            );
            let thread = thread.unwrap();

            // Block on it to retrieve the result
            let mcts = thread.join();
            assert!(
                mcts.is_ok(),
                "Could not join planning thread of {active_agent}! Probably it panicked!"
            );
            let mcts = mcts.unwrap();
            self.executor_state.post_mcts_run_hook(&mcts, active_task);

            // Override the planning task in active_tasks with the best_task we got from the planning
            if log::log_enabled!(log::Level::Info) {
                log::info!(
                    "{} - {} finished planning. Looking for best task...",
                    highlight_tick(tick),
                    highlight_agent(active_agent)
                );
            }
            let best_task = mcts.best_task_with_history(&self.task_history);
            log::info!("Best Task: {best_task:?}");

            self.queue.task_queue.remove(active_task);
            let local_state = D::derive_local_state(&self.state, active_agent);
            let new_active_task =
                self.queue
                    .queue_task(tick, active_agent, best_task.clone(), &local_state);
            self.task_history.insert(active_agent, new_active_task);
        }
    }

    /// Executes all task which are due at the current game tick and starts new planning threads for those agents.
    fn execute_finished_tasks(&mut self, tick: u64) {
        let active_tasks = self.queue.task_queue.clone();
        for active_task in active_tasks.iter().filter(|task| task.end <= tick) {
            // Pop task as it is completed
            self.queue.task_queue.remove(active_task);
            let active_agent = active_task.agent;
            debug_assert!(
                active_task.end == tick,
                "Processing an active task at tick {tick} but it ended at tick {}.",
                active_task.end
            );

            // Should we continue considering that agent?
            if !self
                .executor_state
                .keep_agent(tick, &self.state, active_agent)
            {
                continue;
            }

            // Execute the task, queue the new agents
            let local_state = D::derive_local_state(&self.state, active_agent);
            let (diff, new_task) = self.queue.execute_task(
                active_task,
                &local_state,
                self.executor_state,
                |new_agents_tasks| {
                    for new_task in new_agents_tasks.iter() {
                        self.task_history.insert(new_task.agent, new_task.clone());
                    }
                },
            );
            D::apply(&mut self.state, &local_state, &diff);
            let local_state = D::derive_local_state(&self.state, active_agent);

            // If no next task, spawn a plan task and an associated thread
            let new_task = new_task.unwrap_or_else(|| {
                Box::new(PlanningTask(
                    self.mcts_config.planning_task_duration.unwrap(),
                ))
            });

            // Add new task to queue
            let end_tick = self
                .queue
                .queue_task(tick, active_agent, new_task.clone(), &local_state)
                .end;

            // Deploy new planning thread for this agent if needed
            if new_task.downcast_ref::<PlanningTask>().is_some() {
                let mut mcts = self.new_mcts(tick, active_agent);
                if log::log_enabled!(log::Level::Info) {
                    log::info!(
                        "{} - {} starts planning until {}.",
                        highlight_tick(tick),
                        active_agent,
                        highlight_tick(end_tick)
                    );
                    log::trace!("Active Tasks:");
                    for active_task in &self.queue.task_queue {
                        log::trace!(
                            "{}: {} {:?}",
                            active_task.agent,
                            highlight_tick(active_task.end),
                            active_task.task
                        );
                    }
                }
                let handle = thread::Builder::new()
                    .name(format!("plan-{}", active_agent.0))
                    .spawn(move || {
                        // Initialize MCTS instance for planning
                        // We update it outside the planning thread such that we don't need to pass the state into the thread
                        mcts.run();
                        mcts
                    })
                    .unwrap();
                self.threads.insert(active_task.agent, handle);
            }
        }
    }

    /// Executes all tasks finishing at the current tick and then increments it.
    ///
    /// Returns whether execution should continue.
    pub fn step(&mut self) -> bool {
        if self.queue.is_empty() {
            return false;
        }

        let tick = self.tick.load(Ordering::Relaxed);
        if !self
            .executor_state
            .keep_execution(tick, &self.queue.task_queue, &self.state)
        {
            return false;
        }
        self.block_on_planning(tick);
        self.execute_finished_tasks(tick);
        self.executor_state.post_step_hook(tick, &mut self.state);

        self.tick.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Makes all planning threads stop and wait for them to finish.
    pub fn stop(&mut self) {
        // Set tick to maximum value
        self.tick.store(u64::MAX, Ordering::Relaxed);
        // Wait for planning threads to finish
        self.threads.drain().for_each(|(_, thread)| {
            let _ = thread.join();
        });
    }

    /// Gets the global state, read-only.
    pub fn state(&self) -> &D::GlobalState {
        &self.state
    }

    /// Gets the number of active agents in the execution queue.
    pub fn agents_count(&self) -> usize {
        self.queue.size()
    }
}

/// Creates and runs a multi-threaded executor, initializes state and task queue from the `S` trait.
///
/// Parameter `step_duration` defines how long a logical step lasts in wall time.
pub fn run_threaded_executor<D, S>(
    mcts_config: &MCTSConfiguration,
    executor_state: &mut S,
    step_duration: Duration,
) where
    D: DomainWithPlanningTask + GlobalDomain,
    D::State: Clone + Send,
    D::Diff: Send + Sync,
    S: ExecutorState<D> + ExecutorStateGlobal<D>,
{
    // Initialize the state and the agent queue
    let mut executor = ThreadedExecutor::<D, S>::new(mcts_config.clone(), executor_state);
    loop {
        if !executor.step() {
            break;
        }
        thread::sleep(step_duration);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use core::time;
    use npc_engine_core::{
        ActiveTask, ActiveTasks, AgentId, AgentValue, Behavior, Domain, IdleTask,
        MCTSConfiguration, StateDiffRef, Task,
    };
    use std::{collections::BTreeSet, num::NonZeroU64, thread};

    #[test]
    fn threaded_executor_trivial_domain() {
        #[derive(Debug)]
        enum DisplayAction {
            // #[default] // TODO: use derive(Default) on Rust 1.62 onwards
            Idle,
            Plan,
        }
        impl Default for DisplayAction {
            fn default() -> Self {
                Self::Idle
            }
        }

        struct TrivialDomain;
        impl Domain for TrivialDomain {
            type State = ();
            type Diff = ();
            type DisplayAction = DisplayAction;

            fn list_behaviors() -> &'static [&'static dyn Behavior<Self>] {
                &[&TrivialBehavior]
            }

            fn get_current_value(
                _tick: u64,
                _state_diff: StateDiffRef<Self>,
                _agent: AgentId,
            ) -> AgentValue {
                AgentValue::new(0.).unwrap()
            }

            fn update_visible_agents(
                _start_tick: u64,
                _tick: u64,
                _state_diff: StateDiffRef<Self>,
                agent: AgentId,
                agents: &mut BTreeSet<AgentId>,
            ) {
                agents.insert(agent);
            }

            fn display_action_task_planning() -> Self::DisplayAction {
                DisplayAction::Plan
            }
        }
        impl GlobalDomain for TrivialDomain {
            type GlobalState = ();
            fn derive_local_state(
                _global_state: &Self::GlobalState,
                _agent: AgentId,
            ) -> Self::State {
            }
            fn apply(
                _global_state: &mut Self::GlobalState,
                _local_state: &Self::State,
                _diff: &Self::Diff,
            ) {
            }
        }
        impl DomainWithPlanningTask for TrivialDomain {}

        #[derive(Copy, Clone, Debug)]
        struct TrivialBehavior;
        impl Behavior<TrivialDomain> for TrivialBehavior {
            fn add_own_tasks(
                &self,
                _tick: u64,
                _state_diff: StateDiffRef<TrivialDomain>,
                _agent: AgentId,
                tasks: &mut Vec<Box<dyn Task<TrivialDomain>>>,
            ) {
                tasks.push(Box::new(IdleTask));
            }

            fn is_valid(
                &self,
                _tick: u64,
                _state_diff: StateDiffRef<TrivialDomain>,
                _agent: AgentId,
            ) -> bool {
                true
            }
        }

        struct TrivialExecutorState;
        impl ExecutorStateGlobal<TrivialDomain> for TrivialExecutorState {
            fn create_initial_state(&self) {}
            fn init_task_queue(&self, _: &()) -> ActiveTasks<TrivialDomain> {
                vec![ActiveTask::new_with_end(0, AgentId(0), Box::new(IdleTask))]
                    .into_iter()
                    .collect()
            }
        }
        impl ExecutorState<TrivialDomain> for TrivialExecutorState {}

        env_logger::init();
        let mcts_config = MCTSConfiguration {
            allow_invalid_tasks: false,
            visits: 5,
            depth: 100,
            exploration: 1.414,
            discount_hl: 30.,
            seed: None,
            planning_task_duration: Some(NonZeroU64::new(10).unwrap()),
        };
        let mut executor_state = TrivialExecutorState;
        let mut executor = ThreadedExecutor::new(mcts_config, &mut executor_state);
        let one_millis = time::Duration::from_millis(1);
        for _ in 0..5 {
            executor.step();
            thread::sleep(one_millis);
        }
    }
}
