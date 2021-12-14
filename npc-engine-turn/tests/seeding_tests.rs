use std::{fmt, collections::{BTreeSet, BTreeMap}, hash::{Hasher, Hash}};

use npc_engine_turn::{Domain, Behavior, StateDiffRef, AgentId, Task, StateDiffRefMut, MCTSConfiguration, MCTS};
use rand::{thread_rng, RngCore};

struct TestEngine;

#[derive(Debug, Clone, Copy)]
struct State(usize);

#[derive(Debug, Default, Eq, Hash, Clone, PartialEq)]
struct Diff(usize);

impl Domain for TestEngine {
	type State = State;
	type Diff = Diff;
	type DisplayAction = ();

	fn list_behaviors() -> &'static [&'static dyn Behavior<Self>] {
		&[&TestBehavior]
	}

	fn get_current_value(state_diff: StateDiffRef<Self>, _agent: AgentId) -> f32 {
		state_diff.initial_state.0 as f32 + state_diff.diff.0 as f32
	}

	fn update_visible_agents(_state_diff: StateDiffRef<Self>, agent: AgentId, agents: &mut BTreeSet<AgentId>) {
		agents.insert(agent);
	}
}

#[derive(Copy, Clone, Debug)]
struct TestBehavior;

impl fmt::Display for TestBehavior {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "TestBehavior")
	}
}

impl Behavior<TestEngine> for TestBehavior {
	fn add_own_tasks(
		&self,
		_state_diff: StateDiffRef<TestEngine>,
		_agent: AgentId,
		tasks: &mut Vec<Box<dyn Task<TestEngine>>>,
	) {
		for i in 0..10 {
			tasks.push(Box::new(TestTask(i)) as _);
		}
	}

	fn is_valid(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
		true
	}
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
struct TestTask(usize);

impl fmt::Display for TestTask {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "TestTask({:?})", self.0)
	}
}

impl Task<TestEngine> for TestTask {
	fn weight(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> f32 {
		1.
	}

	fn is_valid(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
		true
	}

	fn execute(
		&self,
		mut state_diff: StateDiffRefMut<TestEngine>,
		_agent: AgentId,
	) -> Option<Box<dyn Task<TestEngine>>> {
		state_diff.diff.0 += self.0.min(1);
		None
	}

	fn display_action(&self) -> <TestEngine as Domain>::DisplayAction {
		()
	}

	fn box_eq(&self, other: &Box<dyn Task<TestEngine>>) -> bool {
		other
			.downcast_ref::<Self>()
			.map(|other| self.eq(other))
			.unwrap_or_default()
	}

	fn box_clone(&self) -> Box<dyn Task<TestEngine>> {
		Box::new(self.clone())
	}

	fn box_hash(&self, mut state: &mut dyn Hasher) {
		self.hash(&mut state)
	}
}

#[test]
fn seed() {
	let agent = AgentId(0);
	for _ in 0..5 {
		let seed = thread_rng().next_u64();
		let config = MCTSConfiguration {
			visits: 1000,
			depth: 10,
			exploration: 1.414,
			discount: 0.95,
			seed: Some(seed),
		};
		let state = State(Default::default());
		let mut mcts = MCTS::<TestEngine>::new(
			state,
			agent,
			&BTreeMap::new(),
			config.clone(),
		);

		let result = mcts.run();

		for _ in 0..10 {
			let mut mcts = MCTS::<TestEngine>::new(
				state,
				agent,
				&BTreeMap::new(),
				config.clone(),
			);

			assert!(result == mcts.run());
		}
	}
}