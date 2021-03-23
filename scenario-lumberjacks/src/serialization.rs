use std::fs;

use crate::{output_path, PreWorldHookArgs, PreWorldHookFn};

pub fn world_serialization_hook() -> PreWorldHookFn {
    Box::new(
        |PreWorldHookArgs {
             world, run, turn, ..
         }| {
            fs::create_dir_all(format!(
                "{}/{}/serialization/",
                output_path(),
                run.map(|n| n.to_string()).unwrap_or_default(),
            ))
            .unwrap();

            let file = fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(format!(
                    "{}/{}/serialization/map{:06}.json",
                    output_path(),
                    run.map(|n| n.to_string()).unwrap_or_default(),
                    turn,
                ))
                .unwrap();

            serde_json::to_writer_pretty(file, world).unwrap();
        },
    )
}
