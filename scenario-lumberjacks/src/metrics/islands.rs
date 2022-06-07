/* 
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich, see AUTHORS.txt for details
 */

use partitions::PartitionVec;

use crate::{PreWorldHookArgs, PreWorldHookFn};

pub fn islands_metric_hook() -> PreWorldHookFn {
    Box::new(|PreWorldHookArgs { world, .. }| {
        let index_fn = |x, y| world.map.width * y + x;

        let mut islands = PartitionVec::with_capacity(world.map.width * world.map.height);
        let mut impassables = 0;

        world.map.tiles.iter().enumerate().for_each(|(y, row)| {
            row.iter().enumerate().for_each(|(x, _tile)| {
                islands.push((x, y));
            });
        });

        world.map.tiles.iter().enumerate().for_each(|(y, row)| {
            row.iter().enumerate().for_each(|(x, tile)| {
                if !tile.is_impassable() {
                    let neighbors = [
                        y.checked_sub(1).map(|y| (x, y)),
                        if y + 1 < world.map.height {
                            Some((x, y + 1))
                        } else {
                            None
                        },
                        x.checked_sub(1).map(|x| (x, y)),
                        if x + 1 < world.map.width {
                            Some((x + 1, y))
                        } else {
                            None
                        },
                    ];

                    let index = index_fn(x, y);
                    neighbors
                        .iter()
                        .cloned()
                        .flatten()
                        .for_each(|(x, y)| {
                            if !world.map.tiles[y][x].is_impassable() {
                                islands.union(index, index_fn(x, y));
                            }
                        });
                } else {
                    impassables += 1;
                }
            });
        });

        // Impassable tiles each count as own island, need to be removed
        println!("# of islands: {}", islands.amount_of_sets() - impassables);
    })
}
