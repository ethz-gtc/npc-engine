/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use npc_engine_core::{debug_name_to_filename_safe, graphviz, ActiveTask, Domain, MCTS};
use std::fs;

/// Plots the MCTS tree using graphviz's dot format.
pub fn plot_tree_in_tmp<D: Domain>(
    mcts: &MCTS<D>,
    base_dir_name: &str,
    file_name: &str,
) -> std::io::Result<()> {
    let temp_dir = std::env::temp_dir().display().to_string();
    let path = format!("{temp_dir}/{base_dir_name}/");
    fs::create_dir_all(&path)?;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(format!("{path}{file_name}.dot"))?;
    graphviz::plot_mcts_tree(mcts, &mut file)
}

/// Plots the MCTS tree using graphviz's dot format, with a filename derived from an active task.
pub fn plot_tree_in_tmp_with_task_name<D: Domain>(
    mcts: &MCTS<D>,
    base_dir_name: &str,
    last_active_task: &ActiveTask<D>,
) -> std::io::Result<()> {
    let time_text = format!("T{}", mcts.start_tick());
    let agent_id_text = format!("A{}", mcts.agent().0);
    let last_task_name = format!("{:?}", last_active_task.task);
    let last_task_name = debug_name_to_filename_safe(&last_task_name);
    plot_tree_in_tmp(
        mcts,
        base_dir_name,
        &format!("{agent_id_text}-{time_text}-{last_task_name}"),
    )
}
