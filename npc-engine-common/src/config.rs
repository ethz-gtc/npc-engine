use std::num::NonZeroU64;

/// A functor that returns whether the planner must do an early stop
pub type EarlyStopCondition = dyn Fn() -> bool + Send;

/// Represents the configuration of an MCTS instance
/// * visits: maximum number of visits per run
/// * depth: maximum tree depth per run
/// * exploration: exploration factor
/// * discount_hl: the discount factor for later reward, in half life (per agent's turn or tick)
/// * seed: optionally, a user-given seed
///
#[derive(Clone, Debug, Default)]
pub struct MCTSConfiguration {
    pub allow_invalid_tasks: bool,
    pub visits: u32,
    pub depth: u32,
    pub exploration: f32,
    pub discount_hl: f32,
    pub planning_task_duration: Option<NonZeroU64>,
    pub seed: Option<u64>,
}

