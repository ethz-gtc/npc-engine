/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use npc_engine_core::{impl_task_boxed_methods, Context, ContextMut, Task, TaskDuration};

use crate::{
    constants::*,
    domain::{DisplayAction, EcosystemDomain},
    map::Tile,
    state::{Access, AccessMut},
};

#[derive(Hash, Clone, Eq, PartialEq, Debug)]
pub struct EatGrass;

impl Task<EcosystemDomain> for EatGrass {
    fn weight(&self, _ctx: Context<EcosystemDomain>) -> f32 {
        EAT_GRASS_WEIGHT
    }

    fn duration(&self, _ctx: Context<EcosystemDomain>) -> TaskDuration {
        0
    }

    fn execute(
        &self,
        mut ctx: ContextMut<EcosystemDomain>,
    ) -> Option<Box<dyn Task<EcosystemDomain>>> {
        let agent_state = ctx.state_diff.get_agent_mut(ctx.agent).unwrap();
        agent_state.food = HERBIVORE_MAX_FOOD;
        let agent_pos = agent_state.position;
        let growth = ctx.state_diff.get_grass(agent_pos).unwrap();
        ctx.state_diff.set_tile(agent_pos, Tile::Grass(growth - 1));
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
        if !agent_state.food < HERBIVORE_MAX_FOOD {
            return false;
        }
        ctx.state_diff
            .get_grass(agent_state.position)
            .filter(|growth| *growth > 0)
            .is_some()
    }

    fn display_action(&self) -> DisplayAction {
        DisplayAction::EatGrass
    }

    impl_task_boxed_methods!(EcosystemDomain);
}
