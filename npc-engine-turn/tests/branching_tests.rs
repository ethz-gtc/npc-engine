use std::{fmt, collections::{BTreeSet, BTreeMap}, hash::{Hasher, Hash}, ops::Range};

use npc_engine_turn::{Domain, Behavior, StateDiffRef, AgentId, Task, StateDiffRefMut, MCTSConfiguration, MCTS};
struct TestEngine;

#[derive(Debug)]
struct State(usize);

#[derive(Debug, Default, Eq, Hash, Clone, PartialEq)]
struct Diff(usize);

impl Domain for TestEngine {
	type State = State;
	type Diff = Diff;
	type DisplayAction = ();

	fn list_behaviors() -> &'static [&'static dyn Behavior<Self>] {
		&[&TestBehaviorA, &TestBehaviorB]
	}

	fn get_current_value(state_diff: StateDiffRef<Self>, _agent: AgentId) -> f32 {
		state_diff.initial_state.0 as f32 + state_diff.diff.0 as f32
	}

	fn update_visible_agents(_state_diff: StateDiffRef<Self>, agent: AgentId, agents: &mut BTreeSet<AgentId>) {
		agents.insert(agent);
	}
}

#[derive(Copy, Clone, Debug)]
struct TestBehaviorA;

impl fmt::Display for TestBehaviorA {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "TestBehaviorA")
	}
}

impl Behavior<TestEngine> for TestBehaviorA {
	fn add_own_tasks(
		&self,
		_state_diff: StateDiffRef<TestEngine>,
		_agent: AgentId,
		tasks: &mut Vec<Box<dyn Task<TestEngine>>>,
	) {
		tasks.push(Box::new(TestTask(true)) as _);
	}

	fn is_valid(&self, _state_diff: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
		true
	}
}

#[derive(Copy, Clone, Debug)]
struct TestBehaviorB;

impl fmt::Display for TestBehaviorB {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "TestBehaviorB")
	}
}

impl Behavior<TestEngine> for TestBehaviorB {
	fn add_own_tasks(
		&self,
		_state_diff: StateDiffRef<TestEngine>,
		_agent: AgentId,
		tasks: &mut Vec<Box<dyn Task<TestEngine>>>,
	) {
		tasks.push(Box::new(TestTask(false)) as _);
	}

	fn is_valid(&self, _state_diff: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
		true
	}
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
struct TestTask(bool);

impl fmt::Display for TestTask {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "TestTask({:?})", self.0)
	}
}

impl Task<TestEngine> for TestTask {
	fn weight(&self, _state_diff: StateDiffRef<TestEngine>, _agent: AgentId) -> f32 {
		1.
	}

	fn is_valid(&self, _state_diff: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
		true
	}

	fn execute(
		&self,
		mut state_diff: StateDiffRefMut<TestEngine>,
		_agent: AgentId,
	) -> Option<Box<dyn Task<TestEngine>>> {
		state_diff.diff.0 += 1;
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

const EPSILON: f32 = 0.001;

#[test]
fn ucb() {
	const CONFIG: MCTSConfiguration = MCTSConfiguration {
		visits: 10,
		depth: 1,
		exploration: 1.414,
		discount: 0.95,
		seed: None
	};
	let agent = AgentId(0);

	let state = State(Default::default());
	let mut mcts = MCTS::<TestEngine>::new(
		state,
		agent,
		&BTreeMap::new(),
		CONFIG
	);

	let task = mcts.run();
	assert!(task.downcast_ref::<TestTask>().is_some());
	// Check length is depth with root
	assert_eq!((CONFIG.depth + 1) as usize, mcts.node_count());

	let node = mcts.root.clone();
	let edges = mcts.get_edges(&node).unwrap();
	let root_visits = edges.child_visits();

	let edge_a = edges
		.expanded_tasks
		.get(&(Box::new(TestTask(true)) as Box<dyn Task<TestEngine>>))
		.unwrap()
		.borrow();
	let edge_b = edges
		.expanded_tasks
		.get(&(Box::new(TestTask(false)) as Box<dyn Task<TestEngine>>))
		.unwrap()
		.borrow();

	assert!(
		(edge_a.uct(
			AgentId(0),
			root_visits,
			CONFIG.exploration,
			Range {
				start: 0.0,
				end: 1.0
			}
		) - 1.9597)
			.abs()
			< EPSILON
	);
	assert!(
		(edge_b.uct(
			AgentId(0),
			root_visits,
			CONFIG.exploration,
			Range {
				start: 0.0,
				end: 1.0
			}
		) - 1.9597)
			.abs()
			< EPSILON
	);
}