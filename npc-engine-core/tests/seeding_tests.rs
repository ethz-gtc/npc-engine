/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::{collections::BTreeSet, fmt, hash::Hash};

use npc_engine_core::{
    impl_task_boxed_methods, AgentId, AgentValue, Behavior, Context, ContextMut, Domain,
    MCTSConfiguration, StateDiffRef, Task, TaskDuration, MCTS,
};
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
struct TestBehavior;

impl Behavior<TestEngine> for TestBehavior {
    fn add_own_tasks(&self, _ctx: Context<TestEngine>, tasks: &mut Vec<Box<dyn Task<TestEngine>>>) {
        for i in 0..10 {
            tasks.push(Box::new(TestTask(i)) as _);
        }
    }

    fn is_valid(&self, _ctx: Context<TestEngine>) -> bool {
        true
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
struct TestTask(u16);

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

    fn execute(&self, mut ctx: ContextMut<TestEngine>) -> Option<Box<dyn Task<TestEngine>>> {
        ctx.state_diff.diff.0 += self.0.min(1);
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
            allow_invalid_tasks: false,
            visits: 1000,
            depth: 10,
            exploration: 1.414,
            discount_hl: 15.,
            seed: Some(seed),
            planning_task_duration: None,
        };
        let state = State(Default::default());
        let mut mcts = MCTS::<TestEngine>::new(state, agent, config.clone());

        let result = mcts.run();

        for _ in 0..10 {
            let mut mcts = MCTS::<TestEngine>::new(state, agent, config.clone());

            assert!(result == mcts.run());
        }
    }
}
