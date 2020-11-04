#[allow(deprecated)]
use std::hash::{BuildHasher, SipHasher};

pub const SEED: u64 = 6364136223846793005;

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

pub(crate) type SeededHashMap<K, V> = std::collections::HashMap<K, V, SeededRandomState>;
pub(crate) type SeededHashSet<V> = std::collections::HashSet<V, SeededRandomState>;
