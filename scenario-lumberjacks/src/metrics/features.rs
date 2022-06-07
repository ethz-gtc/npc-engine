/* 
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use crate::{PreWorldHookArgs, PreWorldHookFn};

pub fn features_metric_hook() -> PreWorldHookFn {
    Box::new(|PreWorldHookArgs { world, .. }| {
        println!("# of trees: {}", world.map.tree_count());
        println!("# of unique patches: {}", world.map.patch_count(2));
    })
}
