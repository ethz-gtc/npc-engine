use std::{fmt, collections::{BTreeSet, BTreeMap}, hash::{Hasher, Hash}};

use npc_engine_turn::{Domain, Behavior, StateDiffRef, AgentId, Task, StateDiffRefMut, MCTSConfiguration, MCTS, impl_task_boxed_methods};

pub(crate) struct TestEngine;

#[derive(Debug)]
pub(crate) struct State(usize);

#[derive(Debug, Default, Eq, Hash, Clone, PartialEq)]
pub(crate) struct Diff(usize);

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

	fn get_current_value(state_diff: StateDiffRef<Self>, _agent: AgentId) -> f32 {
		state_diff.initial_state.0 as f32 + state_diff.diff.0 as f32
	}

	fn update_visible_agents(_state_diff: StateDiffRef<Self>, agent: AgentId, agents: &mut BTreeSet<AgentId>) {
		agents.insert(agent);
	}
}

#[derive(Copy, Clone, Debug)]
struct TestBehavior;

impl Behavior<TestEngine> for TestBehavior {
	fn add_own_tasks(
		&self,
		_state_diff: StateDiffRef<TestEngine>,
		_agent: AgentId,
		tasks: &mut Vec<Box<dyn Task<TestEngine>>>,
	) {
		tasks.push(Box::new(TestTask) as _);
	}

	fn is_valid(&self, _state_diff: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
		true
	}
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
struct TestTask;

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
		DisplayAction
	}

	impl_task_boxed_methods!(TestEngine);
}

const EPSILON: f32 = 0.001;

#[test]
fn linear_bellman() {
	const CONFIG: MCTSConfiguration = MCTSConfiguration {
		visits: 10_000,
		depth: 5,
		exploration: 1.414,
		discount: 0.95,
		seed: None
	};
	let agent = AgentId(0);

	let world = State(0);
	let mut mcts = MCTS::<TestEngine>::new(
		world,
		agent,
		&BTreeMap::new(),
		CONFIG,
	);

	fn expected_value(discount: f32, depth: u32) -> f32 {
		let mut value = 0.;

		for _ in 0..depth {
			value = 1. + discount * value;
		}

		value
	}

	let task = mcts.run();
	assert!(task.downcast_ref::<TestTask>().is_some());
	// Check length is depth with root
	assert_eq!((CONFIG.depth + 1) as usize, mcts.node_count());

	let mut node = mcts.root.clone();

	{
		assert_eq!(Diff(0), node.diff);
	}

	for i in 1..CONFIG.depth {
		let edges = mcts.get_edges(&node).unwrap();
		assert_eq!(edges.expanded_tasks.len(), 1);
		let edge_rc = edges.expanded_tasks.values().nth(0).unwrap();
		let edge = edge_rc.borrow();

		node = edge.child.upgrade().unwrap();

		assert_eq!(Diff(i as usize), node.diff);
		assert_eq!((CONFIG.visits - i + 1) as usize, edge.visits);
		assert!(
			(expected_value(CONFIG.discount, CONFIG.depth - i + 1) - *edge.q_values.get(&agent).unwrap())
				.abs()
				< EPSILON
		);
	}
}
