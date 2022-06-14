/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

#[allow(deprecated)]
use std::hash::{BuildHasher, SipHasher};

/// A seed for seeded hash maps and sets.
const SEED: u64 = 6364136223846793005;

/// An helper struct to carry a seed to HashMaps and HashSets.
#[derive(Copy, Clone, Debug)]
pub struct SeededRandomState {
    seed: u64,
}

impl Default for SeededRandomState {
    fn default() -> Self {
        SeededRandomState { seed: SEED }
    }
}

#[allow(deprecated)]
impl BuildHasher for SeededRandomState {
    type Hasher = SipHasher;

    fn build_hasher(&self) -> Self::Hasher {
        SipHasher::new_with_keys(self.seed, self.seed)
    }
}

/// An `HashMap` with a defined seed.
pub type SeededHashMap<K, V> = std::collections::HashMap<K, V, SeededRandomState>;
/// An `HashSet` with a defined seed.
pub type SeededHashSet<V> = std::collections::HashSet<V, SeededRandomState>;
