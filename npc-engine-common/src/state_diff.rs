use std::{fmt, ops::Deref, mem};

use crate::Domain;

pub struct StateDiffRef<'a, D: Domain> {
    pub initial_state: &'a D::State,
    pub diff: &'a D::Diff,
}
impl<D: Domain> Copy for StateDiffRef<'_, D> {}
impl<D: Domain> Clone for StateDiffRef<'_, D> {
    fn clone(&self) -> Self {
        StateDiffRef::new(self.initial_state, self.diff)
    }
}
impl<'a, D: Domain> StateDiffRef<'a, D> {
    pub fn new(initial_state: &'a D::State, diff: &'a D::Diff) -> Self {
        StateDiffRef { initial_state, diff }
    }
}

impl<D: Domain> fmt::Debug for StateDiffRef<'_, D>
where
    D::State: fmt::Debug,
    D::Diff: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f
            .debug_struct("SnapshotDiffRef")
            .field("Snapshot", self.initial_state)
            .field("Diff", self.diff)
            .finish()
    }
}

pub struct StateDiffRefMut<'a, D: Domain> {
    pub initial_state: &'a D::State,
    pub diff: &'a mut D::Diff,
}
impl<'a, D: Domain> StateDiffRefMut<'a, D> {
    pub fn new(initial_state: &'a D::State, diff: &'a mut D::Diff) -> Self {
        StateDiffRefMut { initial_state, diff }
    }
}

impl<'a, D: Domain> Deref for StateDiffRefMut<'a, D> {
    type Target = StateDiffRef<'a, D>;

    fn deref(&self) -> &Self::Target {
        // Safety: StateDiffRef and StateDiffRefMut have the same memory layout
        // and casting from mutable to immutable is always safe
        unsafe { mem::transmute(self) }
    }
}