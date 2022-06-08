/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::collections::BTreeMap;

use ggez::graphics::Image;
use ggez::Context;
use npc_engine_common::{AgentId, Task, MCTS};

use crate::{Lumberjacks, WorldGlobalState};

pub type PreWorldHookFn = Box<dyn FnMut(PreWorldHookArgs) + 'static>;
pub type PostWorldHookFn = Box<dyn FnMut(PostWorldHookArgs) + 'static>;
pub type PostMCTSHookFn = Box<dyn FnMut(PostMCTSHookArgs) + 'static>;

// Pre world hooks are called once per game loop before any actions have executed
pub struct PreWorldHookArgs<'a, 'b> {
    pub run: Option<usize>,
    pub ctx: &'a mut Option<&'b mut Context>,
    pub assets: &'a BTreeMap<String, Image>,
    pub turn: usize,
    pub world: &'a WorldGlobalState,
}

// Post world hooks are called once per game loop after all actions have executed
pub struct PostWorldHookArgs<'a, 'b> {
    pub run: Option<usize>,
    pub ctx: &'a mut Option<&'b mut Context>,
    pub assets: &'a BTreeMap<String, Image>,
    pub turn: usize,
    pub world: &'a WorldGlobalState,
    pub objectives: &'a BTreeMap<AgentId, Box<dyn Task<Lumberjacks>>>,
}

// Post MCTS hooks are called once per agent per loop after it runs this turn
pub struct PostMCTSHookArgs<'a, 'b> {
    pub run: Option<usize>,
    pub ctx: &'a mut Option<&'b mut Context>,
    pub assets: &'a BTreeMap<String, Image>,
    pub turn: usize,
    pub world: &'a WorldGlobalState,
    pub agent: AgentId,
    pub mcts: &'a MCTS<Lumberjacks>,
    pub objective: Box<dyn Task<Lumberjacks>>,
}
