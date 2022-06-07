/* 
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich, see AUTHORS.txt for details
 */

use crate::{PostMCTSHookArgs, PostMCTSHookFn};

pub fn agency_metric_hook() -> PostMCTSHookFn {
    Box::new(|PostMCTSHookArgs { agent, mcts, .. }| {
        // Agency (diff size)
        // TODO: Only for this agent?
        let (count, size) = mcts
            .nodes()
            .fold((0usize, 0usize), |(count, size), (node, _)| {
                (count + 1, size + node.diff().diff_size())
            });

        println!("Agent{} Agency: {}", agent.0, size as f32 / count as f32);
    })
}
