use std::collections::BTreeMap;

use ggez::graphics::Image;
use ggez::Context;
use npc_engine_turn::{AgentId, Task, MCTS};

use crate::{Lumberjacks, WorldState};

pub type PreWorldHookFn = Box<dyn FnMut(PreWorldHookArgs) + 'static>;
pub type PostWorldHookFn = Box<dyn FnMut(PostWorldHookArgs) + 'static>;
pub type PostMCTSHookFn = Box<dyn FnMut(PostMCTSHookArgs) + 'static>;

// Pre world hooks are called once per game loop before any actions have executed
pub struct PreWorldHookArgs<'a, 'b> {
    pub run: Option<usize>,
    pub ctx: &'a mut Option<&'b mut Context>,
    pub assets: &'a BTreeMap<String, Image>,
    pub turn: usize,
    pub world: &'a WorldState,
}

// Post world hooks are called once per game loop after all actions have executed
pub struct PostWorldHookArgs<'a, 'b> {
    pub run: Option<usize>,
    pub ctx: &'a mut Option<&'b mut Context>,
    pub assets: &'a BTreeMap<String, Image>,
    pub turn: usize,
    pub world: &'a WorldState,
    pub objectives: &'a BTreeMap<AgentId, Box<dyn Task<Lumberjacks>>>,
}

// Post MCTS hooks are called once per agent per loop after it runs this turn
pub struct PostMCTSHookArgs<'a, 'b> {
    pub run: Option<usize>,
    pub ctx: &'a mut Option<&'b mut Context>,
    pub assets: &'a BTreeMap<String, Image>,
    pub turn: usize,
    pub world: &'a WorldState,
    pub agent: AgentId,
    pub mcts: &'a MCTS<Lumberjacks>,
    pub objective: Box<dyn Task<Lumberjacks>>,
}
