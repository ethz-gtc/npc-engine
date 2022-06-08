/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use npc_engine_common::{Domain, StateDiffRef, StateDiffRefMut};
use std::hash::Hash;

/// In case your domain has a `Diff` that is an `Option` of its State,
/// this trait provides generic helper functions.
pub trait OptionDiffDomain {
    type Domain: Domain<State = Self::State, Diff = Option<Self::State>>;
    type State: Clone;
    /// Returns either the `diff` if it is not `None`, or the `initial_state`.
    fn get_cur_state(
        state_diff: StateDiffRef<Self::Domain>,
    ) -> &<<Self as OptionDiffDomain>::Domain as Domain>::State {
        if let Some(diff) = state_diff.diff {
            diff
        } else {
            state_diff.initial_state
        }
    }
    /// Returns either the `diff` if it is not `None`, or copies the `initial_state` into the `diff` and returns it.
    fn get_cur_state_mut(
        state_diff: StateDiffRefMut<Self::Domain>,
    ) -> &mut <<Self as OptionDiffDomain>::Domain as Domain>::State {
        if let Some(diff) = state_diff.diff {
            diff
        } else {
            let diff = state_diff.initial_state.clone();
            *state_diff.diff = Some(diff);
            &mut *state_diff.diff.as_mut().unwrap()
        }
    }
}

impl<
        S: std::fmt::Debug + Sized + Clone + Hash + Eq,
        DA: std::fmt::Debug + Default,
        D: Domain<State = S, Diff = Option<S>, DisplayAction = DA>,
    > OptionDiffDomain for D
{
    type Domain = D;
    type State = <D as Domain>::State;
}
