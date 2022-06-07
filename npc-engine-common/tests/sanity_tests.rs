/* 
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::{collections::BTreeSet, hash::Hash};
use std::fmt;

use npc_engine_common::{Domain, Behavior, StateDiffRef, AgentId, Task, StateDiffRefMut, MCTSConfiguration, MCTS, impl_task_boxed_methods};

#[derive(Debug, Default)]
struct DisplayAction;
impl fmt::Display for DisplayAction {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "")
	}
}

fn init_logger() {
	let _ = env_logger::builder().is_test(true).try_init();
}

mod deferment {
	use npc_engine_common::{AgentValue, TaskDuration};

	use super::*;

	use crate::{Behavior, Task};

	struct TestEngine;

	#[derive(Debug)]
	struct State {
		value: i16,
		investment: i16,
	}

	#[derive(Debug, Default, Eq, Hash, Clone, PartialEq)]
	struct Diff {
		value: i16,
		investment: i16,
	}

	impl Domain for TestEngine {
		type State = State;
		type Diff = Diff;
		type DisplayAction = DisplayAction;

		fn list_behaviors() -> &'static [&'static dyn Behavior<Self>] {
			&[&TestBehavior]
		}

		fn get_current_value(_tick: u64, state: StateDiffRef<Self>, _agent: AgentId) -> AgentValue {
			state.value().into()
		}

		fn update_visible_agents(_start_tick: u64, _tick: u64, _state: StateDiffRef<Self>, agent: AgentId, agents: &mut BTreeSet<AgentId>) {
			agents.insert(agent);
		}
	}

	trait TestState {
		fn value(&self) -> i16;
	}

	trait TestStateMut {
		fn add_value(&mut self, amount: i16);
		fn add_investment(&mut self, amount: i16);
		fn redeem_deferred(&mut self);
	}

	impl TestState for StateDiffRef<'_, TestEngine> {
		fn value(&self) -> i16 {
			self.initial_state.value + self.diff.value
		}
	}

	impl TestStateMut for StateDiffRefMut<'_, TestEngine> {
		fn add_value(&mut self, amount: i16) {
			self.diff.value += amount;
		}

		fn add_investment(&mut self, amount: i16) {
			self.diff.investment += amount;
		}

		fn redeem_deferred(&mut self) {
			self.diff.value += 3 * (self.initial_state.investment + self.diff.investment);
			self.diff.investment = 0 - self.initial_state.investment;
		}
	}

	#[derive(Copy, Clone, Debug)]
	struct TestBehavior;

	impl Behavior<TestEngine> for TestBehavior {
		fn add_own_tasks(
			&self,
			_tick: u64,
			_state: StateDiffRef<TestEngine>,
			_agent: AgentId,
			tasks: &mut Vec<Box<dyn Task<TestEngine>>>,
		) {
			tasks.push(Box::new(TestTaskDirect) as _);
			tasks.push(Box::new(TestTaskDefer) as _);
		}

		fn is_valid(&self, _tick: u64, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
			true
		}
	}

	#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
	struct TestTaskDirect;

	impl Task<TestEngine> for TestTaskDirect {
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
			mut state: StateDiffRefMut<TestEngine>,
			_agent: AgentId,
		) -> Option<Box<dyn Task<TestEngine>>> {
			state.redeem_deferred();
			state.add_value(1);
			None
		}

		fn display_action(&self) -> <TestEngine as Domain>::DisplayAction {
			DisplayAction
		}

		impl_task_boxed_methods!(TestEngine);
	}

	#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
	struct TestTaskDefer;

	impl Task<TestEngine> for TestTaskDefer {
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
			mut state: StateDiffRefMut<TestEngine>,
			_agent: AgentId,
		) -> Option<Box<dyn Task<TestEngine>>> {
			state.redeem_deferred();
			state.add_investment(1);
			None
		}

		fn display_action(&self) -> <TestEngine as Domain>::DisplayAction {
			DisplayAction
		}

		impl_task_boxed_methods!(TestEngine);
	}

	#[test]
	fn deferment() {
		init_logger();
		const CONFIG: MCTSConfiguration = MCTSConfiguration {
			allow_invalid_tasks: false,
			visits: 1000,
			depth: 10,
			exploration: 1.414,
			discount_hl: 15.,
			seed: None,
			planning_task_duration: None,
		};
		init_logger();
		let agent = AgentId(0);

		let state = State {
			value: Default::default(),
			investment: Default::default(),
		};
		let mut mcts = MCTS::<TestEngine>::new(
			state,
			agent,
			CONFIG
		);

		let task = mcts.run().unwrap();
		assert!(task.downcast_ref::<TestTaskDefer>().is_some());
	}
}

mod negative {
	use npc_engine_common::{AgentValue, TaskDuration};

	use super::*;

	use crate::{Behavior, Task};

	struct TestEngine;

	#[derive(Debug)]
	struct State {
		value: i16,
	}

	#[derive(Debug, Default, Eq, Hash, Clone, PartialEq)]
	struct Diff {
		value: i16,
	}

	impl Domain for TestEngine {
		type State = State;
		type Diff = Diff;
		type DisplayAction = DisplayAction;

		fn list_behaviors() -> &'static [&'static dyn Behavior<Self>] {
			&[&TestBehavior]
		}

		fn get_current_value(_tick: u64, state: StateDiffRef<Self>, _agent: AgentId) -> AgentValue {
			state.value().into()
		}

		fn update_visible_agents(_start_tick: u64, _tick: u64, _state: StateDiffRef<Self>, agent: AgentId, agents: &mut BTreeSet<AgentId>) {
			agents.insert(agent);
		}
	}

	trait TestState {
		fn value(&self) -> i16;
	}

	trait TestStateMut {
		fn add_value(&mut self, amount: i16);
	}

	impl TestState for StateDiffRef<'_, TestEngine> {
		fn value(&self) -> i16 {
			self.initial_state.value + self.diff.value
		}
	}

	impl TestStateMut for StateDiffRefMut<'_, TestEngine> {
		fn add_value(&mut self, amount: i16) {
			self.diff.value += amount;
		}
	}

	#[derive(Copy, Clone, Debug)]
	struct TestBehavior;

	impl Behavior<TestEngine> for TestBehavior {
		fn add_own_tasks(
			&self,
			_tick: u64,
			_state: StateDiffRef<TestEngine>,
			_agent: AgentId,
			tasks: &mut Vec<Box<dyn Task<TestEngine>>>,
		) {
			tasks.push(Box::new(TestTaskNoop) as _);
			tasks.push(Box::new(TestTaskNegative) as _);
		}

		fn is_valid(&self, _tick: u64, _state: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
			true
		}
	}

	#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
	struct TestTaskNoop;

	impl Task<TestEngine> for TestTaskNoop {
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
			_state: StateDiffRefMut<TestEngine>,
			_agent: AgentId,
		) -> Option<Box<dyn Task<TestEngine>>> {
			None
		}

		fn display_action(&self) -> <TestEngine as Domain>::DisplayAction {
			DisplayAction
		}

		impl_task_boxed_methods!(TestEngine);
	}

	#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
	struct TestTaskNegative;

	impl Task<TestEngine> for TestTaskNegative {
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
			mut state: StateDiffRefMut<TestEngine>,
			_agent: AgentId,
		) -> Option<Box<dyn Task<TestEngine>>> {
			state.add_value(-1);
			None
		}

		fn display_action(&self) -> <TestEngine as Domain>::DisplayAction {
			DisplayAction
		}

		impl_task_boxed_methods!(TestEngine);
	}

	#[test]
	fn negative() {
		init_logger();
		for depth in (5..=20).step_by(5) {
			let mut noop = 0;
			let mut neg = 0;

			for _ in 0..20 {
				let config = MCTSConfiguration {
					allow_invalid_tasks: false,
					visits: 1.5f32.powi(depth as i32) as u32 / 10 + 100,
					depth,
					exploration: 1.414,
					discount_hl: 15.,
					seed: None,
					planning_task_duration: None,
				};
				let agent = AgentId(0);

				let state = State {
					value: Default::default(),
				};
				let mut mcts = MCTS::<TestEngine>::new(
					state,
					agent,
					config,
				);

				let task = mcts.run().unwrap();
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