use std::num::NonZeroU64;

/// A functor that returns whether the planner must do an early stop.
pub type EarlyStopCondition = dyn Fn() -> bool + Send;

/// The configuration of an MCTS instance.
#[derive(Clone, Debug, Default)]
pub struct MCTSConfiguration {
    /// if true, invalid tasks do not abort expansion or rollout, but trigger re-planning
    pub allow_invalid_tasks: bool,
    /// maximum number of visits per run
    pub visits: u32,
    /// maximum tree depth per run in tick
    pub depth: u32,
    /// exploration factor to use in UCT to balance exploration and exploitation
    pub exploration: f32,
    /// the discount factor for later reward, in half life (per agent's turn or tick)
    pub discount_hl: f32,
    /// if not `None`, the duration of the planning task
    pub planning_task_duration: Option<NonZeroU64>,
    /// optionally, a user-given seed
    pub seed: Option<u64>,
}

