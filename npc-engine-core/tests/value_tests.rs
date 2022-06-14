/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::{collections::BTreeSet, fmt, hash::Hash};

use npc_engine_core::{
    impl_task_boxed_methods, AgentId, AgentValue, Behavior, Domain, MCTSConfiguration,
    StateDiffRef, StateDiffRefMut, Task, TaskDuration, MCTS,
};

pub(crate) struct TestEngine;

#[derive(Debug)]
pub(crate) struct State(u16);

#[derive(Debug, Default, Eq, Hash, Clone, PartialEq)]
pub(crate) struct Diff(u16);

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

    fn get_current_value(
        _tick: u64,
        state_diff: StateDiffRef<Self>,
        _agent: AgentId,
    ) -> AgentValue {
        (state_diff.initial_state.0 + state_diff.diff.0).into()
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
        tasks.push(Box::new(TestTask) as _);
    }

    fn is_valid(&self, _tick: u64, _state_diff: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
        true
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
struct TestTask;

impl Task<TestEngine> for TestTask {
    fn weight(&self, _tick: u64, _state_diff: StateDiffRef<TestEngine>, _agent: AgentId) -> f32 {
        1.
    }

    fn duration(
        &self,
        _tick: u64,
        _state_diff: StateDiffRef<TestEngine>,
        _agent: AgentId,
    ) -> TaskDuration {
        1
    }

    fn is_valid(&self, _tick: u64, _state_diff: StateDiffRef<TestEngine>, _agent: AgentId) -> bool {
        true
    }

    fn execute(
        &self,
        _tick: u64,
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
        allow_invalid_tasks: false,
        visits: 10_000,
        depth: 5,
        exploration: 1.414,
        discount_hl: 15.,
        seed: None,
        planning_task_duration: None,
    };
    env_logger::init();
    let agent = AgentId(0);

    let world = State(0);
    let mut mcts = MCTS::<TestEngine>::new(world, agent, CONFIG);

    fn expected_value(discount: f32, depth: u32) -> f32 {
        let discount = |delta| 2f64.powf((-(delta as f64)) / (discount as f64)) as f32;
        let mut value = 0.;

        for _ in 0..depth {
            value = 1. + discount(1) * value;
        }

        value
    }

    let task = mcts.run().unwrap();
    assert!(task.downcast_ref::<TestTask>().is_some());
    // Check length is depth with root
    assert_eq!((CONFIG.depth + 1) as usize, mcts.node_count());

    let mut node = mcts.root_node();

    {
        assert_eq!(Diff(0), *node.diff());
    }

    for i in 1..CONFIG.depth {
        let edges = mcts.get_edges(&node).unwrap();
        assert_eq!(edges.expanded_count(), 1);
        let edge_rc = edges
            .get_edge(&(Box::new(TestTask) as Box<dyn Task<TestEngine>>))
            .unwrap();
        let edge = edge_rc.lock().unwrap();

        node = edge.child();

        assert_eq!(Diff(i as u16), *node.diff());
        assert_eq!((CONFIG.visits - i + 1) as usize, edge.visits());
        assert!(
            (expected_value(CONFIG.discount_hl, CONFIG.depth - i + 1) - edge.q_value(agent)).abs()
                < EPSILON
        );
    }
}
