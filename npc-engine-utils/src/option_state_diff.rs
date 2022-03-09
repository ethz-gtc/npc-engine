use npc_engine_common::{Domain, StateDiffRef, StateDiffRefMut};

/// In case your domain has a Diff that is an Option of its State,
/// this trait provides generic helper functions.
pub trait OptionDiffDomain {
	type Domain: Domain<State = Self::State, Diff = Option<Self::State>>;
	type State: Clone;
	fn get_cur_state(state_diff: StateDiffRef<Self::Domain>) -> &<<Self as OptionDiffDomain>::Domain as Domain>::State {
		if let Some(diff) = state_diff.diff {
			diff
		} else {
			state_diff.initial_state
		}
	}
	fn get_cur_state_mut(state_diff: StateDiffRefMut<Self::Domain>) -> &mut <<Self as OptionDiffDomain>::Domain as Domain>::State {
		if let Some(diff) = state_diff.diff {
			diff
		} else {
			let diff = state_diff.initial_state.clone();
			*state_diff.diff = Some(diff);
			&mut *state_diff.diff.as_mut().unwrap()
		}
	}
}
