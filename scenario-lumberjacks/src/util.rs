use std::collections::hash_map::DefaultHasher;
use std::f32;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem;

use ggez::graphics::Color;
use num_traits::{AsPrimitive, PrimInt};
use serde::Serialize;

use npc_engine_common::AgentId;

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
    unsafe {
        let bytes: [u8; 8] = mem::transmute(hasher.finish());
        Color::from_rgb(bytes[5], bytes[6], bytes[7])
    }
}

pub const DIRECTIONS: &[Direction] = &[
    Direction::Up,
    Direction::Right,
    Direction::Down,
    Direction::Left,
];

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Direction::Up => write!(f, "Up"),
            Direction::Down => write!(f, "Down"),
            Direction::Left => write!(f, "Left"),
            Direction::Right => write!(f, "Right"),
        }
    }
}

impl Direction {
    pub fn apply(&self, x: isize, y: isize) -> (isize, isize) {
        match self {
            Direction::Up => (x, y - 1),
            Direction::Down => (x, y + 1),
            Direction::Left => (x - 1, y),
            Direction::Right => (x + 1, y),
        }
    }

    pub fn from(start: (isize, isize), end: (isize, isize)) -> Direction {
        match (end.0 - start.0, end.1 - start.1) {
            (1, _) => Direction::Right,
            (-1, _) => Direction::Left,
            (_, 1) => Direction::Down,
            (_, -1) => Direction::Up,
            _ => unreachable!(),
        }
    }
}

#[derive(Copy, Clone, Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Action {
    Walk(Direction),
    Chop(Direction),
    Barrier(Direction),
    Plant(Direction),
    Water(Direction),
    Refill,
    Wait,
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl Default for Action {
    fn default() -> Self {
        Action::Wait
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
