/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::collections::BTreeMap;

use npc_engine_core::{AgentId, MCTSConfiguration, StateDiffRef, StateValueEstimator};
use npc_engine_utils::{NeuralNetwork, Neuron, OptionDiffDomain};

use crate::{domain::LearnDomain, state::State};

#[derive(Clone)]
pub struct NNStateValueEstimator(pub NeuralNetwork<5, 2>);
impl Default for NNStateValueEstimator {
    fn default() -> Self {
        Self(NeuralNetwork {
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
        initial_state: &State,
        _start_tick: u64,
        node: &npc_engine_core::Node<LearnDomain>,
        _edges: &npc_engine_core::Edges<LearnDomain>,
        _depth: u32,
    ) -> Option<BTreeMap<AgentId, f32>> {
        let state = LearnDomain::get_cur_state(StateDiffRef::new(initial_state, node.diff()));
        let value = self.0.output(&state.local_view());
        Some(BTreeMap::from([(AgentId(0), value)]))
    }
}
