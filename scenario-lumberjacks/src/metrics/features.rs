use crate::{PreWorldHookArgs, PreWorldHookFn};

pub fn features_metric_hook() -> PreWorldHookFn {
    Box::new(|PreWorldHookArgs { world, .. }| {
        println!("# of trees: {}", world.map.tree_count());
        println!("# of unique patches: {}", world.map.patch_count(2));
    })
}
