/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::collections::HashMap;

use npc_engine_core::TaskDuration;

use crate::map::{Location, Map};

lazy_static! {
    pub static ref MAP: Map = {
        let links_data = [
            (0, 1, 2),
            (0, 3, 2),
            (1, 2, 2),
            (1, 4, 1),
            (2, 5, 2),
            (3, 4, 1),
            (4, 5, 1),
            (3, 6, 2),
            (5, 6, 2)
        ];
        let capture_locations = [
            0, 2, 6
        ];
        let mut links = HashMap::<Location, HashMap<Location, u64>>::new();
        for (start, end, length) in links_data {
            let start = Location::new(start);
            let end = Location::new(end);
            // add bi-directional link
            links.entry(start)
                .or_default()
                .insert(end, length);
            links.entry(end)
                .or_default()
                .insert(start, length);
        }
        let capture_locations = capture_locations
            .map(Location::new
            )
            .into();
        Map { links, capture_locations }
    };
}

pub const MAX_HP: u8 = 2;
pub const MAX_AMMO: u8 = 4;
pub const CAPTURE_DURATION: TaskDuration = 1;
pub const RESPAWN_AMMO_DURATION: u8 = 2;
pub const RESPAWN_MEDKIT_DURATION: u8 = 17;
