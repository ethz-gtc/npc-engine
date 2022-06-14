/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use constants::{TICKS_PER_ROUND, TOTAL_WOOD};
use domain::LearnDomain;
use estimator::NNStateValueEstimator;
use npc_engine_core::{
    graphviz, ActiveTask, ActiveTasks, AgentId, IdleTask, MCTSConfiguration, StateValueEstimator,
    MCTS,
};
use npc_engine_utils::{run_simple_executor, ExecutorState, ExecutorStateLocal};
use rand::{thread_rng, Rng};
use state::State;

mod behavior;
mod constants;
mod domain;
mod estimator;
mod state;
mod task;

#[derive(Default)]
struct LearnExecutorState {
    estimator: NNStateValueEstimator,
    planned_values: Vec<([f32; 5], f32)>,
}

impl LearnExecutorState {
    pub fn wood_collected(&self) -> f32 {
        TOTAL_WOOD as f32 - self.planned_values.last().unwrap().0.iter().sum::<f32>()
    }

    pub fn train_and_clear_data(&mut self) {
        self.estimator.0.train(self.planned_values.iter(), 0.001);
        self.planned_values.clear();
    }
}

impl ExecutorStateLocal<LearnDomain> for LearnExecutorState {
    fn create_initial_state(&self) -> State {
        let mut rng = thread_rng();
        let mut map = [0; 14];
        for _tree in 0..TOTAL_WOOD {
            let mut pos = rng.gen_range(0..14);
            while map[pos] >= 3 {
                pos = rng.gen_range(0..14);
            }
            map[pos] += 1;
        }
        State {
            map,
            wood_count: 0,
            agent_pos: rng.gen_range(0..14),
        }
    }

    fn init_task_queue(&self, _: &State) -> ActiveTasks<LearnDomain> {
        vec![ActiveTask::new_with_end(0, AgentId(0), Box::new(IdleTask))]
            .into_iter()
            .collect()
    }

    fn keep_agent(&self, tick: u64, _state: &State, _agent: AgentId) -> bool {
        tick < TICKS_PER_ROUND
    }
}

impl ExecutorState<LearnDomain> for LearnExecutorState {
    fn create_state_value_estimator(&self) -> Box<dyn StateValueEstimator<LearnDomain> + Send> {
        Box::new(self.estimator.clone())
    }

    fn post_mcts_run_hook(
        &mut self,
        mcts: &MCTS<LearnDomain>,
        _last_active_task: &ActiveTask<LearnDomain>,
    ) {
        self.planned_values.push((
            mcts.initial_state().local_view(),
            mcts.q_value_at_root(AgentId(0)),
        ));
    }
}

#[allow(dead_code)]
fn enable_map_display() {
    use std::io::Write;
    env_logger::builder()
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .filter(None, log::LevelFilter::Info)
        .init();
}

fn main() {
    // These parameters control the MCTS algorithm.
    const CONFIG: MCTSConfiguration = MCTSConfiguration {
        allow_invalid_tasks: true,
        visits: 20,
        depth: TICKS_PER_ROUND as u32,
        exploration: 1.414,
        discount_hl: TICKS_PER_ROUND as f32 / 3.,
        seed: None,
        planning_task_duration: None,
    };

    // Set the depth of graph output to 4.
    graphviz::set_graph_output_depth(4);

    // Uncomment these if you want to see the world being harvested
    // enable_map_display();

    // State of the execution, including the estimator.
    let mut executor_state = LearnExecutorState::default();

    // We run multiple executions, after each, we train the estimator.
    for _epoch in 0..600 {
        run_simple_executor::<LearnDomain, LearnExecutorState>(&CONFIG, &mut executor_state);
        let wood_collected = executor_state.wood_collected();
        println!("{wood_collected}");
        executor_state.train_and_clear_data();
    }
}
