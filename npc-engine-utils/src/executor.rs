use std::{hash::Hash, sync::{atomic::{AtomicU64, Ordering}, Arc}, collections::HashMap, thread::{JoinHandle, self}};
use npc_engine_common::{Domain, AgentId, Task, ActiveTask, ActiveTasks, StateDiffRef, StateDiffRefMut, MCTS, MCTSConfiguration, IdleTask, StateValueEstimator, DefaultPolicyEstimator, PlanningTask};

/// A domain for which we can run a generic executor
pub trait ExecutableDomain: Domain {
    /// Applies a diff to a mutable state
    fn apply_diff(diff: Self::Diff, state: &mut Self::State);
}
// automatic implementation for domains where diff is an option of state
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
    fn create_state_value_estimator(&self) -> Box<dyn StateValueEstimator<D> + Send> {
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
    /// The tasks of the new agents
    pub new_agents_tasks: Vec<ActiveTask<D>>
}
impl<'a, D, S> SimpleExecutor<'a, D, S>
    where
        D: ExecutableDomain,
        D::State: Clone,
        S: ExecutorState<D>
{
    /// Creates a new executor, initializes state and task queue from the S trait
    pub fn new(mcts_config: MCTSConfiguration, executor_state: &'a mut S) -> Self {
        Self {
            mcts_config,
            state: executor_state.create_initial_state(),
            task_queue: executor_state.init_task_queue(),
            new_agents_tasks: Vec::new(),
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

        // Execute the task and queue the new agents
        let new_task = self.execute_task(&active_task);
        self.queue_new_agents();

        // If no next task, plan and get the task for this agent
        let new_task = new_task.unwrap_or_else(|| {
            log::info!("No subsequent task, planning!");
            let mut mcts = self.new_mcts(tick, active_agent);
            let new_task = mcts.run().unwrap_or_else(|| Box::new(IdleTask));
            self.executor_state.post_mcts_run_hook(&mcts, &active_task);
            new_task
        });

        // Add new task to queue
        self.queue_task(tick, active_agent, new_task);

        true
    }

    fn execute_task(&mut self, active_task: &ActiveTask<D>) -> Option<Box<dyn Task<D>>> {
        let active_agent = active_task.agent;
        let tick = active_task.end;

        // Show state if in debug mode
        let mut diff = D::Diff::default();
        let state_diff = StateDiffRef::<D>::new(&self.state, &diff);
        if log::log_enabled!(log::Level::Info) {
            let highlight_style = ansi_term::Style::new().bold().fg(ansi_term::Colour::Green);
            let tick_text = format!("T{}", tick);
            let task_name = format!("{:?}", active_task.task);
            let agent_id_text = format!("A{}", active_agent.0);
            log::info!("\n{}, State:\n{}\n{} task to be executed: {}", highlight_style.paint(&tick_text), D::get_state_description(state_diff), highlight_style.paint(&agent_id_text), highlight_style.paint(&task_name));
        }

        // Execute task
        let is_task_valid = active_task.task.is_valid(tick, state_diff, active_agent);
        if is_task_valid {
            log::info!("Valid task, executing...");
            let state_diff_mut = StateDiffRefMut::new(&self.state, &mut diff);
            let new_task = active_task.task.execute(tick, state_diff_mut, active_agent);
            self.new_agents_tasks = D::get_new_agents(StateDiffRef::new(&self.state, &diff))
                .into_iter()
                .map(|new_agent| ActiveTask::new_idle(
                    tick,
                    new_agent,
                    active_agent
                ))
                .collect();
            self.executor_state.post_action_execute_hook(&self.state, &diff, &active_task, &mut self.task_queue);
            D::apply_diff(diff, &mut self.state);
            new_task
        } else {
            log::info!("Invalid task!");
            None
        }
    }

    fn queue_new_agents(&mut self) {
        self.task_queue.extend(self.new_agents_tasks.drain(..));
    }

    fn queue_task(&mut self, tick: u64, active_agent: AgentId, new_task: Box<dyn Task<D>>) -> u64 {
        let diff = D::Diff::default();
        let state_diff = StateDiffRef::new(&self.state, &diff);
        if log::log_enabled!(log::Level::Info) {
            let new_active_task = ActiveTask::new(active_agent, new_task.clone(), tick, state_diff);
            log::info!("Queuing new task for A{} until T{}: {:?}", active_agent.0, new_active_task.end, new_task);
        }
        let new_active_task = ActiveTask::new(active_agent, new_task, tick, state_diff);
        let end_tick = new_active_task.end;
        self.task_queue.insert(new_active_task);
        end_tick
    }

    fn new_mcts(&self, tick: u64, active_agent: AgentId) -> MCTS::<D> {
        MCTS::<D>::new_with_tasks(
            self.state.clone(),
            active_agent,
            tick,
            self.task_queue.clone(),
            self.mcts_config.clone(),
            self.executor_state.create_state_value_estimator()
        )
    }
}

/// Creates and runs a single-threaded executor, initializes state and task queue from the S trait
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

/// Domains who want to use the ThreadedExecutor must impement this.
pub trait DomainWithPlanningTask: Domain {
    /// A fallback task, in case, during planning, of the world evolved in a different direction than what the MCTS tree explored
    fn fallback_task(_agent: AgentId) -> Box<dyn Task<Self>> {
        Box::new(IdleTask)
    }
}

/// A multi-threaded generic executor
pub struct ThreadedExecutor<'a, D, S>
where
    D: ExecutableDomain + DomainWithPlanningTask,
    D::State: Clone + Send,
    D::Diff: Send + Sync,
    S: ExecutorState<D>
{
    /// All data of a simple executor are needed as well
    base: SimpleExecutor<'a, D, S>,
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
        D: ExecutableDomain + DomainWithPlanningTask,
        D::State: Clone + Send,
        D::Diff: Send + Sync,
        S: ExecutorState<D>
{
    /// Creates a new executor, initializes state and task queue from the S trait
    pub fn new(mcts_config: MCTSConfiguration, executor_state: &'a mut S) -> Self {
        Self {
            base: SimpleExecutor::new(mcts_config, executor_state),
            task_history: Default::default(),
            threads: Default::default(),
            tick: Arc::new(AtomicU64::new(0)),
        }
    }

    fn find_best_task(&self, mcts: &MCTS<D>) -> Box<dyn Task<D>> {
        let root_agent = mcts.agent();
        log::trace!("Finding best task for {:?} using history", root_agent);
        let current_node = &mcts.root;
        let mut edges = mcts.nodes.get(current_node).unwrap();
        let mut depth = 0;
        loop {
            let node_agent = current_node.agent();
            let node_tick = current_node.tick();
            let edge = if edges.expanded_tasks.len() == 1 {
                let (task, edge) = edges.expanded_tasks.iter().next().unwrap();
                log::trace!("[{depth}][{node_tick}] {node_agent:?} skipping {task:?}");

                // Skip non-branching nodes
                edge
            } else {
                let executed_task = self.task_history.get(&node_agent);
                let executed_task = executed_task
                    .unwrap_or_else(|| panic!("Found no task for {node_agent:?} is history"));
                let task = &executed_task.task;
                log::trace!("[{depth}][{node_tick}] {node_agent:?} executed {task:?}");

                let edge = edges.expanded_tasks.get(task);

                if edge.is_none() {
                    log::debug!("{node_agent:?} executed unexpected {task:?} not present in search tree, returning fallback task");
                    return D::fallback_task(root_agent);
                }

                edge.unwrap()
            };
            let edge = edge.lock().unwrap();
            let current_node = &edge.child.upgrade().unwrap();
            edges = mcts.nodes.get(current_node).unwrap();

            depth += 1;

            // Stop if we reach our own node again
            if current_node.agent() == root_agent {
                break;
            }
        }

        // Return best task, using exploration value of 0
        let range = mcts.min_max_range(mcts.agent());
        let best = edges.best_task(mcts.agent(), 0., range);

        if best.is_none() {
            log::debug!("No valid task for agent {root_agent:?}, returning fallback task");
            return D::fallback_task(root_agent);
        }

        best.unwrap().clone()
    }


    /// Blocks on all planning threads which should have finished in the current tick and adds the
    /// resulting best tasks to the `active_tasks`
    fn block_on_planning(&mut self, tick: u64) {
        // Iterate over all planning tasks that should have finished by now
        let active_tasks = self.base.task_queue.clone();
        for active_task in active_tasks.iter().filter(
            |task| task.end <= tick && task.task.downcast_ref::<PlanningTask>().is_some()
        ) {
            let active_agent = active_task.agent;
            debug_assert!(active_task.end == tick,
                "Processing an active planning task at tick {tick} but it should have been processed at tick {}.", active_task.end
            );

            // Try to get the planning thread of the current agent
            let thread = self.threads.remove(&active_agent);
            assert!(thread.is_some(),
                "There is no planning thread for {active_agent:?} even though there is an active_task for it."
            );
            let thread = thread.unwrap();

            // Block on it to retrieve the result
            let mcts = thread.join();
            assert!(mcts.is_ok(),
                "Could not join planning thread of {active_agent:?}!"
            );
            let mcts = mcts.unwrap();
            self.base.executor_state.post_mcts_run_hook(&mcts, &active_task);

            // #[cfg(feature = "graphviz")]
            //     {
            //         write_graph(&mcts.agent, &mcts, mcts.start_tick);
            //     }

            // Override the planning task in active_tasks with the best_task we got from the planning
            let best_task = self.find_best_task(&mcts);

            if log::log_enabled!(log::Level::Info) {
                let highlight_style = ansi_term::Style::new().bold().fg(ansi_term::Colour::Green);
                let tick_text = format!("T{}", tick);
                log::info!("{} - {:?} finished planning. Best Task: {:?}",
                    highlight_style.paint(&tick_text), active_agent, best_task
                );
            }

            self.base.task_queue.remove(active_task);
            let new_active_task = ActiveTask::new(
                active_agent,
                best_task.clone(),
                tick,
                StateDiffRef::new(&self.base.state, &D::Diff::default()),
            );
            self.base.task_queue.insert(new_active_task.clone());
            self.task_history.insert(active_agent, new_active_task);
        }
    }

    /// Executes all task which are due at the current game tick and starts new planning threads for those agents
    fn execute_finished_tasks(&mut self, tick: u64) {
        let active_tasks = self.base.task_queue.clone();
        for active_task in active_tasks.iter().filter(|task| task.end <= tick) {
            // Pop task as it is completed
            self.base.task_queue.remove(active_task);
            let active_agent = active_task.agent;
            debug_assert!(active_task.end == tick,
                "Processing an active task at tick {tick} but it ended at tick {}.", active_task.end
            );

            // Should we continue considering that agent?
            if !self.base.executor_state.keep_agent(tick, &self.base.state, active_agent) {
                continue;
            }

            // Execute the task, queue the new agents
            let new_task = self.base.execute_task(&active_task);
            for new_task in &self.base.new_agents_tasks {
                self.task_history.insert(new_task.agent, new_task.clone());
            }
            self.base.queue_new_agents();

            // If no next task, spawn a plan task and an associated thread
            let new_task = new_task.unwrap_or_else(||
                Box::new(PlanningTask(self.base.mcts_config.planning_task_duration.unwrap()))
            );

            // Add new task to queue
            let end_tick = self.base.queue_task(tick, active_agent, new_task.clone());

            // Deploy new planning thread for this agent if needed
            if new_task.downcast_ref::<PlanningTask>().is_some() {
                let mut mcts = self.base.new_mcts(tick, active_agent);
                let tick_atomic = self.tick.clone();
                let planning_task_duration = self.base.mcts_config.planning_task_duration
                    .expect("Planning task must have non-zero duration for threaded executor");
                mcts.early_stop_condition = Some(Box::new(move || {
                    tick_atomic.load(Ordering::Relaxed) >= tick + planning_task_duration.get() - 1
                }));
                if log::log_enabled!(log::Level::Info) {
                    let highlight_style = ansi_term::Style::new().bold().fg(ansi_term::Colour::Green);
                    let tick_text = format!("T{}", tick);
                    let end_tick_text = format!("T{}", end_tick);
                    log::info!("{} - {:?} starts planning until {}.",
                        highlight_style.paint(&tick_text), active_agent, highlight_style.paint(&end_tick_text)
                    );
                    log::trace!("Active Tasks:");
                    for active_task in &self.base.task_queue {
                        let end_tick_text = format!("T{}", active_task.end);
                        log::trace!("{:?}: {} {:?}",
                            active_task.agent, highlight_style.paint(&end_tick_text), active_task.task
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
                    }).unwrap();
                self.threads.insert(active_task.agent, handle);
            }
        }
    }

    /// Executes one task, returns whether there are still tasks in the queue
    pub fn step(&mut self) -> bool {
        if self.base.task_queue.is_empty() {
            return false;
        }

        let tick = self.tick.load(Ordering::Relaxed);
        self.block_on_planning(tick);
        self.execute_finished_tasks(tick);

        self.tick.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Make all planning threads stop and wait for them to finish
    pub fn stop(&mut self) {
        // Set tick to maximum value
        self.tick.store(u64::MAX, Ordering::Relaxed);
        // Wait for planning threads to finish
        self.threads.drain().for_each(|(_, thread)| {
            let _ = thread.join();
        });
    }
}

#[cfg(test)]
mod tests {
    use core::time;
    use std::{collections::BTreeSet, thread, num::NonZeroU64};
    use crate::*;
    use npc_engine_common::{Domain, Behavior, Task, AgentId, AgentValue, StateDiffRef, ActiveTask, ActiveTasks, IdleTask, MCTSConfiguration};

    #[test]
    fn threaded_executor_trivial_domain() {

        #[derive(Debug)]
        enum DisplayAction {
            // #[default] // TODO: use derive(Default) on Rust 1.62 unwards
            Idle,
            Plan
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
        
            fn get_current_value(_tick: u64, _state_diff: StateDiffRef<Self>, _agent: AgentId) -> AgentValue {
                AgentValue::new(0.).unwrap()
            }
        
            fn update_visible_agents(_start_tick: u64, _tick: u64, _state_diff: StateDiffRef<Self>, agent: AgentId, agents: &mut BTreeSet<AgentId>) {
                agents.insert(agent);
            }

            fn display_action_task_planning() -> Self::DisplayAction {
                DisplayAction::Plan
            }
        }
        impl ExecutableDomain for TrivialDomain {
            fn apply_diff(_diff: Self::Diff, _state: &mut Self::State) { }
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
        
            fn is_valid(&self, _tick: u64, _state_diff: StateDiffRef<TrivialDomain>, _agent: AgentId) -> bool {
                true
            }
        }

        struct TrivialExecutorState;
        impl ExecutorState<TrivialDomain> for TrivialExecutorState {
            fn create_initial_state(&self) -> () {
                ()
            }

            fn init_task_queue(&self) -> ActiveTasks<TrivialDomain> {
                vec![
                    ActiveTask::new_with_end(0, AgentId(0), Box::new(IdleTask)),
                ].into_iter().collect()
            }

            fn keep_agent(&self, _tick: u64, _state: &(), _agent: AgentId) -> bool {
                true
            }
        }

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
        let mut executor = ThreadedExecutor::new(
            mcts_config,
            &mut executor_state
        );
        let one_millis = time::Duration::from_millis(1);
        for _ in 0..5 {
            executor.step();
            thread::sleep(one_millis);
        }
    }
}