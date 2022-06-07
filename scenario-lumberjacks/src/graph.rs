/* 
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich, see AUTHORS.txt for details
 */

use std::fs;

use npc_engine_common::graphviz::plot_mcts_tree;

use crate::{output_path, PostMCTSHookArgs, PostMCTSHookFn};

pub fn graph_hook() -> PostMCTSHookFn {
    Box::new(
        |PostMCTSHookArgs {
             run,
             turn,
             agent,
             mcts,
             ..
         }| {
            fs::create_dir_all(format!(
                "{}/{}/graphs/agent{}/",
                output_path(),
                run.map(|n| n.to_string()).unwrap_or_default(),
                agent.0
            ))
            .unwrap();
            let mut file = fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(format!(
                    "{}/{}/graphs/agent{}/turn{:06}.dot",
                    output_path(),
                    run.map(|n| n.to_string()).unwrap_or_default(),
                    agent.0,
                    turn
                ))
                .unwrap();

            plot_mcts_tree(mcts, &mut file).unwrap();
        },
    )
}
