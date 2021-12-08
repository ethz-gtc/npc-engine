/// A functor that returns whether the planner must do an early stop
pub type EarlyStopCondition = dyn Fn() -> bool;

/// Represents the configuration of an MCTS instance
/// * visits: maximum number of visits per run
/// * depth: maximum tree depth per run
/// * exploration: exploration factor
/// * discount: the discounter factor (per agent's turn or tick)
/// * seed: optionally, a user-given seed
///
#[derive(Clone, Debug)]
pub struct MCTSConfiguration {
    pub visits: u32,
    pub depth: u32,
    pub exploration: f32,
    pub discount: f32,
    pub seed: Option<u64>,
}

