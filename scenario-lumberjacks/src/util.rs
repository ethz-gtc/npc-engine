/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::collections::hash_map::DefaultHasher;
use std::f32;
use std::fmt;
use std::hash::{Hash, Hasher};

use ggez::graphics::Color;
use npc_engine_utils::Coord2D;
use npc_engine_utils::Direction;
use npc_engine_utils::DirectionConverterYDown;
use num_traits::{AsPrimitive, PrimInt};
use serde::Serialize;

use npc_engine_core::AgentId;

pub const SPRITE_SIZE: f32 = 32.;

#[derive(Clone, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct Map2D<T> {
    width: usize,
    tiles: Box<[T]>,
}

impl<T: fmt::Debug> fmt::Debug for Map2D<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for y in 0..self.height() {
            write!(f, " ")?;
            for x in 0..self.width() {
                write!(f, "{:?} ", self.get(x, y).unwrap())?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl<T> Map2D<T> {
    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.tiles.len() / self.width
    }

    pub fn new_square(size: usize, value: T) -> Self
    where
        T: Clone,
    {
        Self::new(size, size, value)
    }

    pub fn new(width: usize, height: usize, value: T) -> Self
    where
        T: Clone,
    {
        Map2D {
            width,
            tiles: vec![value; width * height].into_boxed_slice(),
        }
    }

    fn index<SX, SY>(&self, x: SX, y: SY) -> Option<usize>
    where
        SX: PrimInt + AsPrimitive<usize>,
        SY: PrimInt + AsPrimitive<usize>,
    {
        if x >= SX::zero() && x.as_() < self.width() && y >= SY::zero() && y.as_() < self.height() {
            let x = x.as_();
            let y = y.as_();
            Some(y * self.width + x)
        } else {
            None
        }
    }

    pub fn get<SX, SY>(&self, x: SX, y: SY) -> Option<&T>
    where
        SX: PrimInt + AsPrimitive<usize>,
        SY: PrimInt + AsPrimitive<usize>,
    {
        self.index(x, y).map(|index| &self.tiles[index])
    }

    pub fn get_mut<SX, SY>(&mut self, x: SX, y: SY) -> Option<&mut T>
    where
        SX: PrimInt + AsPrimitive<usize>,
        SY: PrimInt + AsPrimitive<usize>,
    {
        self.index(x, y).map(move |index| &mut self.tiles[index])
    }
}

pub fn agent_color(agent: AgentId) -> Color {
    let mut hasher = DefaultHasher::default();
    agent.0.hash(&mut hasher);
    let bytes: [u8; 8] = hasher.finish().to_ne_bytes();
    Color::from_rgb(bytes[5], bytes[6], bytes[7])
}

pub fn apply_direction(direction: Direction, x: isize, y: isize) -> (isize, isize) {
    let pos = DirectionConverterYDown::apply(direction, Coord2D::new(x as i32, y as i32));
    (pos.x as isize, pos.y as isize)
}

pub fn from_direction(start: (isize, isize), end: (isize, isize)) -> Direction {
    let start = Coord2D::new(start.0 as i32, start.1 as i32);
    let end = Coord2D::new(end.0 as i32, end.1 as i32);
    DirectionConverterYDown::from(start, end)
}

#[derive(Copy, Clone, Debug, Default, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Action {
    Walk(Direction),
    Chop(Direction),
    Barrier(Direction),
    Plant(Direction),
    Water(Direction),
    Refill,
    #[default]
    Wait,
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Action {
    pub fn sprite_name(&self) -> &str {
        match self {
            Action::Wait => "Right",
            Action::Walk(Direction::Up) => "Top",
            Action::Walk(Direction::Down) => "Down",
            Action::Walk(Direction::Left) => "Left",
            Action::Walk(Direction::Right) => "Right",
            Action::Chop(Direction::Up) => "TopChopping",
            Action::Chop(Direction::Down) => "DownChopping",
            Action::Chop(Direction::Left) => "LeftChopping",
            Action::Chop(Direction::Right) => "RightChopping",
            Action::Barrier(Direction::Up) => "TopBarrier",
            Action::Barrier(Direction::Down) => "DownBarrier",
            Action::Barrier(Direction::Left) => "LeftBarrier",
            Action::Barrier(Direction::Right) => "RightBarrier",
            Action::Plant(Direction::Up) => "TopBarrier",
            Action::Plant(Direction::Down) => "DownBarrier",
            Action::Plant(Direction::Left) => "LeftBarrier",
            Action::Plant(Direction::Right) => "RightBarrier",
            Action::Refill => "Right",
            Action::Water(Direction::Up) => "TopBarrier",
            Action::Water(Direction::Down) => "DownBarrier",
            Action::Water(Direction::Left) => "LeftBarrier",
            Action::Water(Direction::Right) => "RightBarrier",
        }
    }
}
