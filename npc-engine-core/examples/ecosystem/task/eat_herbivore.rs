/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

// use std::fmt::{self, Formatter};

use npc_engine_core::{impl_task_boxed_methods, Context, ContextMut, Task, TaskDuration};
use npc_engine_utils::Direction;

use crate::{
    constants::*,
    domain::{DisplayAction, EcosystemDomain},
    map::DirConv,
    state::{Access, AccessMut, AgentType},
};

#[derive(Hash, Clone, Eq, PartialEq, Debug)]
pub struct EatHerbivore(pub Direction);

impl Task<EcosystemDomain> for EatHerbivore {
    fn weight(&self, _ctx: Context<EcosystemDomain>) -> f32 {
        EAT_HERBIVORE_WEIGHT
    }

    fn duration(&self, _ctx: Context<EcosystemDomain>) -> TaskDuration {
        0
    }

    fn execute(
        &self,
        mut ctx: ContextMut<EcosystemDomain>,
    ) -> Option<Box<dyn Task<EcosystemDomain>>> {
        let agent_state = ctx.state_diff.get_agent(ctx.agent).unwrap();
        // try next to position
        let passage_pos = DirConv::apply(self.0, agent_state.position);
        let prey_state = ctx.state_diff.get_agent_at_mut(passage_pos);
        if let Some((_, prey_state)) = prey_state {
            if prey_state.ty == AgentType::Herbivore && prey_state.alive() {
                prey_state.kill(ctx.tick);
                let agent_state = ctx.state_diff.get_agent_mut(ctx.agent).unwrap();
                agent_state.position = passage_pos;
                agent_state.food = CARNIVORE_MAX_FOOD;
                return None;
            }
        }
        // if not, the prey is one further away
        let target_pos = DirConv::apply(self.0, passage_pos);
        let prey_state = ctx.state_diff.get_agent_at_mut(target_pos).unwrap();
        prey_state.1.kill(ctx.tick);
        let agent_state = ctx.state_diff.get_agent_mut(ctx.agent).unwrap();
        agent_state.position = target_pos;
        agent_state.food = CARNIVORE_MAX_FOOD;
        None
    }

    fn is_valid(&self, ctx: Context<EcosystemDomain>) -> bool {
        let agent_state = ctx.state_diff.get_agent(ctx.agent).unwrap();
        debug_assert!(
            agent_state.alive(),
            "Task validity check called on a dead agent"
        );
        if !agent_state.alive() {
            return false;
        }
        if !agent_state.food < CARNIVORE_MAX_FOOD {
            return false;
        }
        let is_herbivore = |position| {
            ctx.state_diff.is_tile_passable(position)
                && ctx
                    .state_diff
                    .get_agent_at(position)
                    .map(|(_, agent_state)| {
                        agent_state.ty == AgentType::Herbivore && agent_state.alive()
                    })
                    .unwrap_or(false)
        };
        let passage_pos = DirConv::apply(self.0, agent_state.position);
        let target_pos = DirConv::apply(self.0, passage_pos);
        is_herbivore(passage_pos)
            || (ctx.state_diff.is_tile_passable(passage_pos) && is_herbivore(target_pos))
    }

    fn display_action(&self) -> DisplayAction {
        DisplayAction::EatHerbivore(self.0)
    }

    impl_task_boxed_methods!(EcosystemDomain);
}
