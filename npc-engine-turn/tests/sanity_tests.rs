use std::{collections::{BTreeSet, BTreeMap}, hash::{Hasher, Hash}};

use npc_engine_turn::{Domain, Behavior, StateDiffRef, AgentId, Task, StateDiffRefMut, MCTSConfiguration, MCTS};

mod deferment {
	use std::fmt;

	use super::*;

	use crate::{Behavior, Task};

	struct TestEngine;

	#[derive(Debug)]
	struct State {
		value: isize,
		investment: isize,
	}

	#[derive(Debug, Default, Eq, Hash, Clone, PartialEq)]
	struct Diff {
		value: isize,
		investment: isize,
	}

	impl Domain for TestEngine {
		type State = State;
		type Diff = Diff;
		type DisplayAction = ();

		fn list_behaviors() -> &'static [&'static dyn Behavior<Self>] {
			&[&TestBehavior]
		}

		fn get_current_value(state: StateDiffRef<Self>, _agent: AgentId) -> f32 {
			state.value()
		}

		fn update_visible_agents(_state: StateDiffRef<Self>, agent: AgentId, agents: &mut BTreeSet<AgentId>) {
			agents.insert(agent);
		}
	}

	trait TestState {
		fn value(&self) -> f32;
	}

	trait TestStateMut {
		fn add_value(&mut self, amount: isize);
		fn add_investment(&mut self, amount: isize);
		fn redeem_deferred(&mut self);
	}

	impl TestState for StateDiffRef<'_, TestEngine> {
		fn value(&self) -> f32 {
			self.initial_state.value as f32 + self.diff.value as f32
		}
	}

	impl TestStateMut for StateDiffRefMut<'_, TestEngine> {
		fn add_value(&mut self, amount: isize) {
			self.diff.value += amount;
		}

		fn add_investment(&mut self, amount: isize) {
			self.diff.investment += amount;
		}

		fn redeem_deferred(&mut self) {
			self.diff.value += 3 * (self.initial_state.investment + self.diff.investment);
			self.diff.investment = 0 - self.initial_state.investment;
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
			_state: StateDiffRef<TestEngine>,
			_agent: AgentId,
			tasks: &mut Vec<Box<dyn Task<TestEngine>>>,
		) {
			tasks.push(Box::new(TestTaskDirect) as _);
			tasks.push(Box::new(TestTaskDefer) as _);
		}

		fn is_valid(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
			true
		}
	}

	#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
	struct TestTaskDirect;

	impl fmt::Display for TestTaskDirect {
		fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
			write!(f, "TestTaskDirect")
		}
	}

	impl Task<TestEngine> for TestTaskDirect {
		fn weight(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> f32 {
			1.
		}

		fn is_valid(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
			true
		}

		fn execute(
			&self,
			mut state: StateDiffRefMut<TestEngine>,
			_agent: AgentId,
		) -> Option<Box<dyn Task<TestEngine>>> {
			state.redeem_deferred();
			state.add_value(1);
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

	#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
	struct TestTaskDefer;

	impl fmt::Display for TestTaskDefer {
		fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
			write!(f, "TestTaskDefer")
		}
	}

	impl Task<TestEngine> for TestTaskDefer {
		fn weight(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> f32 {
			1.
		}

		fn is_valid(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
			true
		}

		fn execute(
			&self,
			mut state: StateDiffRefMut<TestEngine>,
			_agent: AgentId,
		) -> Option<Box<dyn Task<TestEngine>>> {
			state.redeem_deferred();
			state.add_investment(1);
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
	fn deferment() {
		const CONFIG: MCTSConfiguration = MCTSConfiguration {
			visits: 1000,
			depth: 10,
			exploration: 1.414,
			discount: 0.95,
			seed: None
		};
		let agent = AgentId(0);

		let state = State {
			value: Default::default(),
			investment: Default::default(),
		};
		let mut mcts = MCTS::<TestEngine>::new(
			state,
			agent,
			&BTreeMap::new(),
			CONFIG
		);

		let task = mcts.run();
		assert!(task.downcast_ref::<TestTaskDefer>().is_some());
	}
}

mod negative {
	use std::fmt;

	use super::*;

	use crate::{Behavior, Task};

	struct TestEngine;

	#[derive(Debug)]
	struct State {
		value: isize,
	}

	#[derive(Debug, Default, Eq, Hash, Clone, PartialEq)]
	struct Diff {
		value: isize,
	}

	impl Domain for TestEngine {
		type State = State;
		type Diff = Diff;
		type DisplayAction = ();

		fn list_behaviors() -> &'static [&'static dyn Behavior<Self>] {
			&[&TestBehavior]
		}

		fn get_current_value(state: StateDiffRef<Self>, _agent: AgentId) -> f32 {
			state.value()
		}

		fn update_visible_agents(_state: StateDiffRef<Self>, agent: AgentId, agents: &mut BTreeSet<AgentId>) {
			agents.insert(agent);
		}

		fn get_visible_agents(state: StateDiffRef<Self>, agent: AgentId) -> BTreeSet<AgentId> {
			let mut agents = BTreeSet::new();
			Self::update_visible_agents(state, agent, &mut agents);
			agents
		}

		fn get_tasks<'a>(
			state: StateDiffRef<'a, Self>,
			agent: AgentId
		) -> Vec<Box<dyn Task<Self>>> {
			let mut actions = Vec::new();
			Self::list_behaviors()
				.iter()
				.filter(|behavior| behavior.is_valid(state, agent))
				.for_each(|behavior| behavior.add_tasks(state, agent, &mut actions));

			actions.dedup();
			actions
		}
	}

	trait TestState {
		fn value(&self) -> f32;
	}

	trait TestStateMut {
		fn add_value(&mut self, amount: isize);
	}

	impl TestState for StateDiffRef<'_, TestEngine> {
		fn value(&self) -> f32 {
			self.initial_state.value as f32 + self.diff.value as f32
		}
	}

	impl TestStateMut for StateDiffRefMut<'_, TestEngine> {
		fn add_value(&mut self, amount: isize) {
			self.diff.value += amount;
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
			_state: StateDiffRef<TestEngine>,
			_agent: AgentId,
			tasks: &mut Vec<Box<dyn Task<TestEngine>>>,
		) {
			tasks.push(Box::new(TestTaskNoop) as _);
			tasks.push(Box::new(TestTaskNegative) as _);
		}

		fn is_valid(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
			true
		}
	}

	#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
	struct TestTaskNoop;

	impl fmt::Display for TestTaskNoop {
		fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
			write!(f, "TestTaskNoop")
		}
	}

	impl Task<TestEngine> for TestTaskNoop {
		fn weight(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> f32 {
			1.
		}

		fn is_valid(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
			true
		}

		fn execute(
			&self,
			_state: StateDiffRefMut<TestEngine>,
			_agent: AgentId,
		) -> Option<Box<dyn Task<TestEngine>>> {
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

	#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
	struct TestTaskNegative;

	impl fmt::Display for TestTaskNegative {
		fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
			write!(f, "TestTaskNegative")
		}
	}

	impl Task<TestEngine> for TestTaskNegative {
		fn weight(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> f32 {
			1.
		}

		fn is_valid(&self, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
			true
		}

		fn execute(
			&self,
			mut state: StateDiffRefMut<TestEngine>,
			_agent: AgentId,
		) -> Option<Box<dyn Task<TestEngine>>> {
			state.add_value(-1);
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
	fn negative() {
		for depth in (5..=20).step_by(5) {
			let mut noop = 0;
			let mut neg = 0;

			for _ in 0..20 {
				let config = MCTSConfiguration {
					visits: 1.5f32.powi(depth as i32) as u32 / 10 + 100,
					depth,
					exploration: 1.414,
					discount: 0.95,
					seed: None
				};
				let agent = AgentId(0);

				let state = State {
					value: Default::default(),
				};
				let mut mcts = MCTS::<TestEngine>::new(
					state,
					agent,
					&BTreeMap::new(),
					config,
				);

				let task = mcts.run();
				if task.downcast_ref::<TestTaskNoop>().is_some() {
					noop += 1;
				} else {
					neg += 1;
				}
			}

			assert!((noop as f32 / (noop + neg) as f32) >= 0.75);
		}
	}
}