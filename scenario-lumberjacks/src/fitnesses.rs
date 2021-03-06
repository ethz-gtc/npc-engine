/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use crate::{Lumberjacks, WorldState};
use npc_engine_core::{AgentId, StateDiffRef};

pub(crate) fn minimalist(state: StateDiffRef<Lumberjacks>, _agent: AgentId) -> f32 {
    -(state.trees().len() as f32)
}
