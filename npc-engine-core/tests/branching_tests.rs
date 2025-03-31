/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::{collections::BTreeSet, fmt, hash::Hash, ops::Range};

use npc_engine_core::{
    impl_task_boxed_methods, AgentId, AgentValue, Behavior, Context, ContextMut, Domain,
    MCTSConfiguration, StateDiffRef, Task, TaskDuration, MCTS,
};
struct TestEngine;

#[derive(Debug)]
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
        &[&TestBehaviorA, &TestBehaviorB]
    }

    fn get_current_value(
        _tick: u64,
        state_diff: StateDiffRef<Self>,
        _agent: AgentId,
    ) -> AgentValue {
        (state_diff.initial_state.0 + state_diff.diff.0).into()
    }

    fn update_visible_agents(
        _start_tick: u64,
        ctx: Context<TestEngine>,
        agents: &mut BTreeSet<AgentId>,
    ) {
        agents.insert(ctx.agent);
    }
}

#[derive(Copy, Clone, Debug)]
struct TestBehaviorA;

impl Behavior<TestEngine> for TestBehaviorA {
    fn add_own_tasks(&self, _ctx: Context<TestEngine>, tasks: &mut Vec<Box<dyn Task<TestEngine>>>) {
        tasks.push(Box::new(TestTask(true)) as _);
    }

    fn is_valid(&self, _ctx: Context<TestEngine>) -> bool {
        true
    }
}

#[derive(Copy, Clone, Debug)]
struct TestBehaviorB;

impl Behavior<TestEngine> for TestBehaviorB {
    fn add_own_tasks(&self, _ctx: Context<TestEngine>, tasks: &mut Vec<Box<dyn Task<TestEngine>>>) {
        tasks.push(Box::new(TestTask(false)) as _);
    }

    fn is_valid(&self, _ctx: Context<TestEngine>) -> bool {
        true
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
struct TestTask(bool);

impl Task<TestEngine> for TestTask {
    fn weight(&self, _ctx: Context<TestEngine>) -> f32 {
        1.
    }

    fn duration(&self, _ctx: Context<TestEngine>) -> TaskDuration {
        1
    }

    fn is_valid(&self, _ctx: Context<TestEngine>) -> bool {
        true
    }

    fn execute(&self, ctx: ContextMut<TestEngine>) -> Option<Box<dyn Task<TestEngine>>> {
        ctx.state_diff.diff.0 += 1;
        None
    }

    fn display_action(&self) -> <TestEngine as Domain>::DisplayAction {
        DisplayAction
    }

    impl_task_boxed_methods!(TestEngine);
}

const EPSILON: f32 = 0.001;

#[test]
fn ucb() {
    const CONFIG: MCTSConfiguration = MCTSConfiguration {
        allow_invalid_tasks: false,
        visits: 10,
        depth: 1,
        exploration: 1.414,
        discount_hl: 15.,
        seed: None,
        planning_task_duration: None,
    };
    env_logger::init();
    let agent = AgentId(0);

    let state = State(Default::default());
    let mut mcts = MCTS::<TestEngine>::new(state, agent, CONFIG);

    let task = mcts.run().unwrap();
    assert!(task.downcast_ref::<TestTask>().is_some());
    assert_eq!((CONFIG.depth * 2 + 1) as usize, mcts.node_count());

    let node = mcts.root_node();
    let edges = mcts.get_edges(&node).unwrap();
    let root_visits = edges.child_visits();

    let edge_a = edges
        .get_edge(&(Box::new(TestTask(true)) as Box<dyn Task<TestEngine>>))
        .unwrap();
    let edge_a = edge_a.lock().unwrap();
    let edge_b = edges
        .get_edge(&(Box::new(TestTask(false)) as Box<dyn Task<TestEngine>>))
        .unwrap();
    let edge_b = edge_b.lock().unwrap();

    assert!(
        (edge_a.uct(
            AgentId(0),
            root_visits,
            CONFIG.exploration,
            Range {
                start: AgentValue::new(0.0).unwrap(),
                end: AgentValue::new(1.0).unwrap()
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
                start: AgentValue::new(0.0).unwrap(),
                end: AgentValue::new(1.0).unwrap()
            }
        ) - 1.9597)
            .abs()
            < EPSILON
    );
}
