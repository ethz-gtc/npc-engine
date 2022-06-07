/* 
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use npc_engine_common::{Domain, AgentId};


/// A domain that provides a global state, out of which a local state for the planning can be derived.
pub trait GlobalDomain: Domain {
	/// Global state: all data that can change in the course of the simulation.
    type GlobalState: std::fmt::Debug + Sized + 'static;

	/// Derives a new local state for the given agent from the given global state.
    fn derive_local_state(global_state: &Self::GlobalState, agent: AgentId) -> Self::State;

    /// Applies a diff from a local state to the global state.
    fn apply(global_state: &mut Self::GlobalState, local_state: &Self::State, diff: &Self::Diff);
}