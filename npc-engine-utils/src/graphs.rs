use std::fs;
use npc_engine_common::{Domain, MCTS, graphviz};

pub fn plot_tree_in_tmp<D: Domain>(mcts: &MCTS::<D>, base_dir_name: &str, file_name: &str) -> std::io::Result<()> {
	let temp_dir = std::env::temp_dir().display().to_string();
	let path = format!("{temp_dir}/{base_dir_name}/");
	fs::create_dir_all(&path)?;
	let mut file = fs::OpenOptions::new()
		.create(true)
		.write(true)
		.truncate(true)
		.open(
			format!("{path}{file_name}.dot")
		)?;
		graphviz::plot_mcts_tree(mcts, &mut file)
}