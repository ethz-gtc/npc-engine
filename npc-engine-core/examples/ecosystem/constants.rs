/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

// This file contains constants that can be tuned in order to adjust simulation.

use std::time::Duration;

use npc_engine_core::TaskDuration;
use npc_engine_utils::Coord2D;

// map generation
pub const MAP_SIZE: Coord2D = Coord2D::new(40, 20);
pub const OBSTACLE_RANDOM_COUNT: usize = 14;
pub const OBSTACLE_HOTSPOT_COUNT: usize = 6;
pub const PLANT_RANDOM_COUNT: usize = 50;
pub const PLANT_HOTSPOT_COUNT: usize = 77;
pub const HERBIVORE_COUNT: usize = 40;
pub const CARNIVORE_COUNT: usize = 8;

// local state derivation
pub const MAP_RADIUS: i32 = 8;
pub const AGENTS_RADIUS_HERBIVORE: i32 = 3;
pub const AGENTS_RADIUS_CARNIVORE: i32 = 6;
pub const MAX_AGENTS_ATTENTION: usize = 3; // the number of closest agents an agent considers

// food parameter
pub const HERBIVORE_MAX_FOOD: u32 = 6;
pub const CARNIVORE_MAX_FOOD: u32 = 16;

// reproduction parameters
pub const HERBIVORE_REPRODUCTION_PERIOD: u64 = 5;
pub const HERBIVORE_REPRODUCTION_CYCLE_DURATION: u64 =
    WORLD_TASK_DURATION * HERBIVORE_REPRODUCTION_PERIOD;
pub const HERBIVORE_MIN_FOOD_FOR_REPRODUCTION: u32 = 4;
pub const HERBIVORE_FOOD_GIVEN_TO_BABY: u32 = 2;
pub const CARNIVORE_REPRODUCTION_PERIOD: u64 = 7;
pub const CARNIVORE_REPRODUCTION_CYCLE_DURATION: u64 =
    WORLD_TASK_DURATION * CARNIVORE_REPRODUCTION_PERIOD;
pub const CARNIVORE_MIN_FOOD_FOR_REPRODUCTION: u32 = 12;
pub const CARNIVORE_FOOD_GIVEN_TO_BABY: u32 = 5;

// world task parameter
pub const WORLD_TASK_DURATION: TaskDuration = 10; // each worl task duraction, the food is reduced by one
pub const WORLD_GRASS_REGROW_PERIOD: u64 = 7;
pub const WORLD_GRASS_GROW_CYCLE_DURATION: u64 = WORLD_TASK_DURATION * WORLD_GRASS_REGROW_PERIOD;
pub const WORLD_GRASS_EXPAND_PERIOD: u64 = 13;
pub const WORLD_GRASS_EXPAND_CYCLE_DURATION: u64 = WORLD_TASK_DURATION * WORLD_GRASS_EXPAND_PERIOD;

// death
pub const TOMB_DURATION: u64 = 40; // number of tick a tombstone is shown after death

// planning parameters
pub const PLANNING_DURATION: u64 = 3;
pub const PLANNING_VISITS: u32 = 1000;
pub const PLANNING_MINIMUM_VISITS: u32 = 50;
pub const PLANNING_DEPTH: u32 = 50;
pub const PLANNING_DISCOUNT_HL: f32 = PLANNING_DEPTH as f32 / 3.0;
pub const PLANNING_EXPLORATION: f32 = 1.414;

// task weights (idle task has weight 1)
pub const EAT_GRASS_WEIGHT: f32 = 5.0;
pub const EAT_HERBIVORE_WEIGHT: f32 = 20.0;
pub const JUMP_WEIGHT: f32 = 10.0;
pub const MOVE_WEIGHT: f32 = 5.0;

// execution paramters
pub const EXECUTION_STEP_DURATION: Duration = Duration::from_millis(40);
