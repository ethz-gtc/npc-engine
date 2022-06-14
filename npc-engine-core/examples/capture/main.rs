/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::{collections::BTreeMap, iter};

use behavior::world::WORLD_AGENT_ID;
use constants::MAX_HP;
use domain::CaptureDomain;
use map::Location;
use npc_engine_core::{
    graphviz, ActiveTask, ActiveTasks, AgentId, IdleTask, MCTSConfiguration, MCTS,
};
use npc_engine_utils::{
    plot_tree_in_tmp_with_task_name, run_simple_executor, ExecutorState, ExecutorStateLocal,
};
use state::{AgentState, CapturePointState, State};
use task::world::WorldStep;

#[macro_use]
extern crate lazy_static;

mod behavior;
mod constants;
mod domain;
mod map;
mod state;
mod task;

struct CaptureGameExecutorState;
impl ExecutorStateLocal<CaptureDomain> for CaptureGameExecutorState {
    fn create_initial_state(&self) -> State {
        let agent0_id = AgentId(0);
        let agent0_state = AgentState {
            acc_capture: 0,
            cur_or_last_location: Location::new(0),
            next_location: None,
            hp: MAX_HP,
            ammo: 0, //MAX_AMMO,
        };
        let agent1_id = AgentId(1);
        let agent1_state = AgentState {
            acc_capture: 0,
            cur_or_last_location: Location::new(6),
            next_location: None,
            hp: MAX_HP,
            ammo: 0, //MAX_AMMO,
        };
        State {
            agents: BTreeMap::from([(agent0_id, agent0_state), (agent1_id, agent1_state)]),
            capture_points: [
                CapturePointState::Free,
                CapturePointState::Free,
                CapturePointState::Free,
            ],
            ammo: 1,
            ammo_tick: 0,
            medkit: 1,
            medkit_tick: 0,
        }
    }

    fn init_task_queue(&self, state: &State) -> ActiveTasks<CaptureDomain> {
        state
            .agents
            .iter()
            .map(|(id, _)| ActiveTask::new_with_end(0, *id, Box::new(IdleTask)))
            .chain(iter::once(ActiveTask::new_with_end(
                0,
                WORLD_AGENT_ID,
                Box::new(WorldStep),
            )))
            .collect()
    }

    fn keep_agent(&self, _tick: u64, state: &State, agent: AgentId) -> bool {
        agent == WORLD_AGENT_ID || state.agents.contains_key(&agent)
    }
}

impl ExecutorState<CaptureDomain> for CaptureGameExecutorState {
    fn post_mcts_run_hook(
        &mut self,
        mcts: &MCTS<CaptureDomain>,
        last_active_task: &ActiveTask<CaptureDomain>,
    ) {
        if let Err(e) = plot_tree_in_tmp_with_task_name(mcts, "capture_graphs", last_active_task) {
            println!("Cannot write search tree: {e}");
        }
    }
}

fn main() {
    // These parameters control the MCTS algorithm.
    const CONFIG: MCTSConfiguration = MCTSConfiguration {
        allow_invalid_tasks: true,
        visits: 5000,
        depth: 50,
        exploration: 1.414,
        discount_hl: 17.,
        seed: None,
        planning_task_duration: None,
    };

    // Set the depth of graph output to 7.
    graphviz::set_graph_output_depth(7);

    // Configure the long to just write its content and enable the info level.
    use std::io::Write;
    env_logger::builder()
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .filter(None, log::LevelFilter::Info)
        .init();

    // State of the execution.
    let mut executor_state = CaptureGameExecutorState;

    // Run the execution.
    run_simple_executor::<CaptureDomain, CaptureGameExecutorState>(&CONFIG, &mut executor_state);
}
