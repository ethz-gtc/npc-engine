use std::{fmt, collections::BTreeSet, hash::Hash};

use npc_engine_common::{Domain, Behavior, StateDiffRef, AgentId, Task, StateDiffRefMut, MCTSConfiguration, MCTS, impl_task_boxed_methods, AgentValue, TaskDuration};
use rand::{thread_rng, RngCore};

struct TestEngine;

#[derive(Debug, Clone, Copy)]
struct State(u16);

#[derive(Debug, Default, Eq, Hash, Clone, PartialEq)]
struct Diff(u16);

#[derive(Debug, Default)]
struct DisplayAction;
impl fmt::Display for DisplayAction {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "")
	}
}

impl Domain for TestEngine {
	type State = State;
	type Diff = Diff;
	type DisplayAction = DisplayAction;

	fn list_behaviors() -> &'static [&'static dyn Behavior<Self>] {
		&[&TestBehavior]
	}

	fn get_current_value(_tick: u64, state_diff: StateDiffRef<Self>, _agent: AgentId) -> AgentValue {
		(state_diff.initial_state.0 + state_diff.diff.0).into()
	}

	fn update_visible_agents(_tick: u64, _state_diff: StateDiffRef<Self>, agent: AgentId, agents: &mut BTreeSet<AgentId>) {
		agents.insert(agent);
	}
}

#[derive(Copy, Clone, Debug)]
struct TestBehavior;

impl Behavior<TestEngine> for TestBehavior {
	fn add_own_tasks(
		&self,
		_tick: u64,
		_state_diff: StateDiffRef<TestEngine>,
		_agent: AgentId,
		tasks: &mut Vec<Box<dyn Task<TestEngine>>>,
	) {
		for i in 0..10 {
			tasks.push(Box::new(TestTask(i)) as _);
		}
	}

	fn is_valid(&self, _tick: u64, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
		true
	}
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
struct TestTask(u16);

impl Task<TestEngine> for TestTask {
	fn weight(&self, _tick: u64, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> f32 {
		1.
	}

	fn duration(&self, _tick: u64, _state_diff: StateDiffRef<TestEngine>, _agent: AgentId) -> TaskDuration {
		1
    }

	fn is_valid(&self, _tick: u64, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
		true
	}

	fn execute(
		&self,
		_tick: u64,
		mut state_diff: StateDiffRefMut<TestEngine>,
		_agent: AgentId,
	) -> Option<Box<dyn Task<TestEngine>>> {
		state_diff.diff.0 += self.0.min(1);
		None
	}

	fn display_action(&self) -> <TestEngine as Domain>::DisplayAction {
		DisplayAction
	}

	impl_task_boxed_methods!(TestEngine);
}

#[test]
fn seed() {
	env_logger::init();
	let agent = AgentId(0);
	for _ in 0..5 {
		let seed = thread_rng().next_u64();
		let config = MCTSConfiguration {
			visits: 1000,
			depth: 10,
			exploration: 1.414,
			discount_hl: 15.,
			seed: Some(seed),
		};
		let state = State(Default::default());
		let mut mcts = MCTS::<TestEngine>::new(
			state,
			agent,
			config.clone(),
		);

		let result = mcts.run();

		for _ in 0..10 {
			let mut mcts = MCTS::<TestEngine>::new(
				state,
				agent,
				config.clone(),
			);

			assert!(result == mcts.run());
		}
	}
}