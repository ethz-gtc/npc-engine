/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::collections::HashMap;

use nonmax::NonMaxU8;

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
pub struct Location(NonMaxU8);
impl Location {
    pub fn new(location: u8) -> Self {
        Self(NonMaxU8::new(location).unwrap())
    }
    pub fn get(&self) -> NonMaxU8 {
        self.0
    }
}

pub struct Map {
    pub links: HashMap<Location, HashMap<Location, u64>>,
    pub capture_locations: Vec<Location>,
}
impl Map {
    pub fn ammo_location(&self) -> Location {
        Location::new(1)
    }
    pub fn medkit_location(&self) -> Location {
        Location::new(5)
    }
    pub fn path_len(&self, from: Location, to: Location) -> Option<u64> {
        self.links
            .get(&from)
            .and_then(|ends| ends.get(&to).copied())
    }
    pub fn neighbors(&self, from: Location) -> Vec<Location> {
        self.links
            .get(&from)
            .map_or(Vec::new(), |ends| ends.keys().copied().collect())
    }
    pub fn is_path(&self, from: Location, to: Location) -> bool {
        self.path_len(from, to).is_some()
    }
    pub fn capture_location(&self, index: u8) -> Location {
        self.capture_locations[index as usize]
    }
    pub fn capture_locations_count(&self) -> u8 {
        self.capture_locations.len() as u8
    }
}
