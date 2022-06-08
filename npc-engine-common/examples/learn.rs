/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::collections::BTreeMap;

use npc_engine_common::{
    graphviz, impl_task_boxed_methods, ActiveTask, ActiveTasks, AgentId, AgentValue, Behavior,
    Domain, IdleTask, MCTSConfiguration, StateDiffRef, StateDiffRefMut, StateValueEstimator, Task,
    TaskDuration, MCTS,
};
use npc_engine_utils::{
    run_simple_executor, ExecutorState, ExecutorStateLocal, NetworkWithHiddenLayer, Neuron,
    OptionDiffDomain,
};
use rand::{thread_rng, Rng};

const TOTAL_WOOD: usize = 20;
const TICKS_PER_ROUND: u64 = 50;

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
struct State {
    map: [u8; 14],
    wood_count: u8,
    agent_pos: u8,
}
impl State {
    // The number of trees as seen by the agent:
    // [sum very left, just left, cur pos, just right, sum very right]
    fn local_view(&self) -> [f32; 5] {
        let pos = self.agent_pos as usize;
        let len = self.map.len();
        let left_left = if pos > 1 {
            let sum: u8 = self.map.iter().take(pos - 1).sum();
            sum as f32
        } else {
            0.
        };
        let left = if pos > 0 {
            self.map[pos - 1] as f32
        } else {
            0.
        };
        let mid = self.map[pos] as f32;
        let right = if pos < len - 1 {
            self.map[pos + 1] as f32
        } else {
            0.
        };
        let right_right = if pos < len - 2 {
            let sum: u8 = self.map.iter().skip(pos + 2).sum();
            sum as f32
        } else {
            0.
        };
        [left_left, left, mid, right, right_right]
    }
}
type Diff = Option<State>; // if Some, use this diff, otherwise use initial state

#[derive(Debug)]
enum DisplayAction {
    Wait,
    Collect,
    Left,
    Right,
}
impl Default for DisplayAction {
    fn default() -> Self {
        Self::Wait
    }
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
struct Collect;
impl Task<LearnDomain> for Collect {
    fn duration(
        &self,
        _tick: u64,
        _state_diff: StateDiffRef<LearnDomain>,
        _agent: AgentId,
    ) -> TaskDuration {
        1
    }

    fn execute(
        &self,
        _tick: u64,
        state_diff: StateDiffRefMut<LearnDomain>,
        _agent: AgentId,
    ) -> Option<Box<dyn Task<LearnDomain>>> {
        let state = LearnDomain::get_cur_state_mut(state_diff);
        debug_assert!(state.map[state.agent_pos as usize] > 0);
        state.map[state.agent_pos as usize] -= 1;
        state.wood_count += 1;
        None
    }

    fn is_valid(&self, _tick: u64, state_diff: StateDiffRef<LearnDomain>, _agent: AgentId) -> bool {
        let state = LearnDomain::get_cur_state(state_diff);
        state.map[state.agent_pos as usize] > 0
    }

    fn display_action(&self) -> DisplayAction {
        DisplayAction::Collect
    }

    impl_task_boxed_methods!(LearnDomain);
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
struct Left;
impl Task<LearnDomain> for Left {
    fn duration(
        &self,
        _tick: u64,
        _state_diff: StateDiffRef<LearnDomain>,
        _agent: AgentId,
    ) -> TaskDuration {
        1
    }

    fn execute(
        &self,
        _tick: u64,
        state_diff: StateDiffRefMut<LearnDomain>,
        _agent: AgentId,
    ) -> Option<Box<dyn Task<LearnDomain>>> {
        let state = LearnDomain::get_cur_state_mut(state_diff);
        debug_assert!(state.agent_pos > 0);
        state.agent_pos -= 1;
        None
    }

    fn is_valid(&self, _tick: u64, state_diff: StateDiffRef<LearnDomain>, _agent: AgentId) -> bool {
        let state = LearnDomain::get_cur_state(state_diff);
        state.agent_pos > 0
    }

    fn display_action(&self) -> DisplayAction {
        DisplayAction::Left
    }

    impl_task_boxed_methods!(LearnDomain);
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
struct Right;
impl Task<LearnDomain> for Right {
    fn duration(
        &self,
        _tick: u64,
        _state_diff: StateDiffRef<LearnDomain>,
        _agent: AgentId,
    ) -> TaskDuration {
        1
    }

    fn execute(
        &self,
        _tick: u64,
        state_diff: StateDiffRefMut<LearnDomain>,
        _agent: AgentId,
    ) -> Option<Box<dyn Task<LearnDomain>>> {
        let state = LearnDomain::get_cur_state_mut(state_diff);
        debug_assert!((state.agent_pos as usize) < state.map.len() - 1);
        state.agent_pos += 1;
        None
    }

    fn is_valid(&self, _tick: u64, state_diff: StateDiffRef<LearnDomain>, _agent: AgentId) -> bool {
        let state = LearnDomain::get_cur_state(state_diff);
        (state.agent_pos as usize) < state.map.len() - 1
    }

    fn display_action(&self) -> DisplayAction {
        DisplayAction::Right
    }

    impl_task_boxed_methods!(LearnDomain);
}

struct DefaultBehaviour;
impl Behavior<LearnDomain> for DefaultBehaviour {
    fn add_own_tasks(
        &self,
        tick: u64,
        state_diff: StateDiffRef<LearnDomain>,
        agent: AgentId,
        tasks: &mut Vec<Box<dyn Task<LearnDomain>>>,
    ) {
        tasks.push(Box::new(IdleTask));
        let possible_tasks: [Box<dyn Task<LearnDomain>>; 3] =
            [Box::new(Collect), Box::new(Left), Box::new(Right)];
        for task in &possible_tasks {
            if task.is_valid(tick, state_diff, agent) {
                tasks.push(task.clone());
            }
        }
    }

    fn is_valid(
        &self,
        _tick: u64,
        _state_diff: StateDiffRef<LearnDomain>,
        _agent: AgentId,
    ) -> bool {
        true
    }
}

struct LearnDomain;
impl Domain for LearnDomain {
    type State = State;
    type Diff = Diff;
    type DisplayAction = DisplayAction;

    fn list_behaviors() -> &'static [&'static dyn Behavior<Self>] {
        &[&DefaultBehaviour]
    }

    fn get_current_value(
        _tick: u64,
        state_diff: StateDiffRef<Self>,
        _agent: AgentId,
    ) -> AgentValue {
        let state = Self::get_cur_state(state_diff);
        AgentValue::from(state.wood_count)
    }

    fn update_visible_agents(
        _start_tick: u64,
        _tick: u64,
        _state_diff: StateDiffRef<Self>,
        agent: AgentId,
        agents: &mut std::collections::BTreeSet<AgentId>,
    ) {
        agents.insert(agent);
    }

    fn get_state_description(state_diff: StateDiffRef<Self>) -> String {
        let state = Self::get_cur_state(state_diff);
        let map = state
            .map
            .iter()
            .map(|count| match count {
                0 => "  ",
                1 => "🌱",
                2 => "🌿",
                _ => "🌳",
            })
            .collect::<String>();
        format!("@{:2} 🪵{:2} [{map}]", state.agent_pos, state.wood_count)
    }
}

#[derive(Clone)]
struct NNStateValueEstimator(NetworkWithHiddenLayer<5, 2>);
impl Default for NNStateValueEstimator {
    fn default() -> Self {
        Self(NetworkWithHiddenLayer {
            hidden_layer: [
                Neuron::random_with_range(0.1),
                Neuron::random_with_range(0.1),
            ],
            output_layer: Neuron::random_with_range(0.1),
        })
    }
}
impl StateValueEstimator<LearnDomain> for NNStateValueEstimator {
    fn estimate(
        &mut self,
        _rnd: &mut rand_chacha::ChaCha8Rng,
        _config: &MCTSConfiguration,
        initial_state: &<LearnDomain as Domain>::State,
        _start_tick: u64,
        node: &npc_engine_common::Node<LearnDomain>,
        _edges: &npc_engine_common::Edges<LearnDomain>,
        _depth: u32,
    ) -> Option<BTreeMap<AgentId, f32>> {
        let state = LearnDomain::get_cur_state(StateDiffRef::new(initial_state, node.diff()));
        let value = self.0.output(&state.local_view());
        Some(BTreeMap::from([(AgentId(0), value)]))
    }
}

#[derive(Default)]
struct LearnExecutorState {
    estimator: NNStateValueEstimator,
    planned_values: Vec<([f32; 5], f32)>,
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
            mcts.initial_state.local_view(),
            mcts.q_value_at_root(AgentId(0)),
        ));
    }
}

fn main() {
    const CONFIG: MCTSConfiguration = MCTSConfiguration {
        allow_invalid_tasks: true,
        visits: 20,
        depth: TICKS_PER_ROUND as u32,
        exploration: 1.414,
        discount_hl: TICKS_PER_ROUND as f32 / 3.,
        seed: None,
        planning_task_duration: None,
    };
    graphviz::set_graph_output_depth(4);
    /*use std::io::Write;
    env_logger::builder()
        .format(|buf, record|
            writeln!(buf, "{}", record.args())
        )
        .filter(None, log::LevelFilter::Info)
        .init();*/
    let mut executor_state = LearnExecutorState::default();
    for _epoch in 0..600 {
        run_simple_executor::<LearnDomain, LearnExecutorState>(&CONFIG, &mut executor_state);
        let wood_collected = TOTAL_WOOD as f32
            - executor_state
                .planned_values
                .last()
                .unwrap()
                .0
                .iter()
                .sum::<f32>();
        // println!("{epoch}: collected {wood_collected}");
        println!("{wood_collected}");
        executor_state
            .estimator
            .0
            .train(executor_state.planned_values.iter(), 0.001);
        executor_state.planned_values.clear();
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn state_local_view() {
        let mut state = State {
            map: [1, 3, 2, 1, 3, 2, 1, 0, 1, 3, 2, 0, 1, 3],
            wood_count: 0,
            agent_pos: 0,
        };
        assert_eq!(state.local_view(), [0., 0., 1., 3., 19.]);
        state.agent_pos = 1;
        assert_eq!(state.local_view(), [0., 1., 3., 2., 17.]);
        state.agent_pos = 3;
        assert_eq!(state.local_view(), [4., 2., 1., 3., 13.]);
        state.agent_pos = 12;
        assert_eq!(state.local_view(), [19., 0., 1., 3., 0.]);
        state.agent_pos = 13;
        assert_eq!(state.local_view(), [19., 1., 3., 0., 0.]);
    }
}
