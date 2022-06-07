/* 
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::{fmt, str::FromStr};

use npc_engine_utils::{Coord2D, DirectionConverterYDown};

#[derive(Debug, Clone)]
pub struct ParseTileError;
impl fmt::Display for ParseTileError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "invalid tile character")
    }
}

#[derive(Debug, Hash, Copy, Clone, Eq, PartialEq)]
pub enum Tile {
    Grass(u8),
    Obstacle,
}
impl Tile {
    pub fn is_passable(&self) -> bool {
		*self != Tile::Obstacle
	}
}
impl FromStr for Tile {
	type Err = ParseTileError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
		match s {
			"#" => Ok(Self::Obstacle),
			s => match s.parse::<u8>() {
				Ok(v) => match v {
					0..=3 => Ok(Tile::Grass(v)),
					_ => Err(ParseTileError)
				},
				Err(_) => Err(ParseTileError),
			}
		}
	}
}

pub trait GridAccess {
	type Tile: Clone;

	fn new(size: Coord2D, tile: Tile) -> Self;

	fn size(&self) -> Coord2D;

	fn at(&self, coord: Coord2D) -> Option<&Self::Tile>;

	fn at_mut(&mut self, coord: Coord2D) -> Option<&mut Self::Tile>;

	fn extract_region(&self, center: Coord2D, extent: i32) -> (Coord2D, Self) where Self: Sized {
		let extent = Coord2D::new(extent, extent);
		let top_left = center - extent;
		let origin = top_left.max_per_comp(Coord2D::new(0, 0));
		let bottom_right = center + Coord2D::new(1, 1) + extent;
		let size = bottom_right.min_per_comp(self.size()) - origin;
		let mut map = Self::new(size, Tile::Obstacle);
		assert!(size.x > 0);
		assert!(size.y > 0);
		for y in 0..size.y {
			for x in 0..size.x {
				let local = Coord2D::new(x, y);
				*map.at_mut(local).unwrap() = self.at(local + origin).unwrap().clone();
			}
		}
		(origin, map)
	}
}

#[derive(Debug, Clone)]
pub enum ParseGridError<T: FromStr> {
	TileError(T::Err),
	InconsistentLines
}
fn parse_map_str<T: FromStr>(s: &str) -> Result<Box<[Box<[T]>]>, ParseGridError<T>> {
	let mut map = Vec::new();
	let mut previous_length: Option<usize> = None;
	for line in s.split("\n") {
		if previous_length.map_or(false, |len| len != line.len()) {
			return Err(ParseGridError::InconsistentLines);
		}
		previous_length = Some(line.len());
		map.push(line.chars()
			.map(|c| match T::from_str(&c.to_string()) {
				Ok(tile) => Ok(tile),
				Err(e) => Err(ParseGridError::TileError(e)),
			})
			.collect::<Result<Vec<_>, _>>()?
			.into_boxed_slice()
		)
	}
	Ok(map.into_boxed_slice())
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Map(pub Box<[Box<[Tile]>]>);
impl Map {
	pub fn empty() -> Self {
		Self(vec![Vec::new().into_boxed_slice()].into_boxed_slice())
	}
}

impl GridAccess for Map {
	type Tile = Tile;

	fn new(size: Coord2D, tile: Tile) -> Self {
		let width = size.x;
		let height = size.y;
		assert!(width > 0);
		assert!(height > 0);
		let mut map = Vec::new();
		for _y in 0..height {
			map.push((0..width)
				.map(|_| tile)
				.collect::<Vec<_>>()
				.into_boxed_slice()
			);
		}
		Self(map.into_boxed_slice())
	}

	fn size(&self) -> Coord2D {
		let height = self.0.len();
		let width = self.0[0].len();
		Coord2D::new(width as i32, height as i32)
	}

    fn at(&self, coord: Coord2D) -> Option<&Tile> {
		let x: usize = TryInto::try_into(coord.x).ok()?;
		let y: usize = TryInto::try_into(coord.y).ok()?;
		self.0.get(y)
			.and_then(|value| value.get(x))
	}

    fn at_mut(&mut self, coord: Coord2D) -> Option<&mut Tile> {
		let x: usize = TryInto::try_into(coord.x).ok()?;
		let y: usize = TryInto::try_into(coord.y).ok()?;
        self.0.get_mut(y)
			.and_then(|value| value.get_mut(x))
    }
}

impl FromStr for Map {
    type Err = ParseGridError<Tile>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
		Ok(Map(parse_map_str(s)?))
    }
}

pub type DirConv = DirectionConverterYDown;

#[cfg(test)]
mod tests {
    use npc_engine_utils::Coord2D;

    use crate::map::{Tile, GridAccess};
    use super::Map;
	use std::str::FromStr;

	#[test]
	fn passable() {
		assert!(Tile::Grass(0).is_passable());
		assert!(Tile::Grass(1).is_passable());
		assert!(Tile::Grass(2).is_passable());
		assert!(Tile::Grass(3).is_passable());
		assert!(!Tile::Obstacle.is_passable());
	}

	#[test]
	fn map_new_and_size() {
		let size = Coord2D::new(4, 3);
		let map = Map::new(size, Tile::Grass(2));
		assert_eq!(map.size(), size);
	}

    #[test]
	fn map_access() {
		// build map
		let mut map = Map::from_str(
			"#0000\n\
			 01230\n\
			 ###00"
		).unwrap();
		// check content
		let at_checked = |x, y| map.at(Coord2D::new(x, y));
		let at = |x, y| *at_checked(x, y).unwrap();
		assert_eq!(at(0, 0), Tile::Obstacle);
		assert_eq!(at(3, 0), Tile::Grass(0));
		assert_eq!(at(0, 2), Tile::Obstacle);
		assert_eq!(at(4, 0), Tile::Grass(0));
		assert_eq!(at(0, 1), Tile::Grass(0));
		assert_eq!(at(1, 1), Tile::Grass(1));
		assert_eq!(at(2, 1), Tile::Grass(2));
		assert_eq!(at(3, 1), Tile::Grass(3));
		assert_eq!(at_checked(-1, -1), None);
		assert_eq!(at_checked(6, 0), None);
		assert_eq!(at_checked(0, 4), None);
		// do some changes
		*map.at_mut(Coord2D::new(3, 0)).unwrap() = Tile::Obstacle;
		*map.at_mut(Coord2D::new(0, 2)).unwrap() = Tile::Grass(3);
		// check changed content
		let at_checked = |x, y| map.at(Coord2D::new(x, y));
		let at = |x, y| *at_checked(x, y).unwrap();
		assert_eq!(at(3, 0), Tile::Obstacle);
		assert_eq!(at(0, 2), Tile::Grass(3));
	}

	#[test]
	fn map_extract_region() {
		let map = Map::from_str(
			"#0000#\n\
			 012302\n\
			 ###003"
		).unwrap();
		let extract = map.extract_region(Coord2D::new(0, 0), 1);
		assert_eq!(extract.0, Coord2D::new(0, 0));
		assert_eq!(extract.1,
			Map::from_str(
				"#0\n\
				 01"
			).unwrap()
		);
		let extract = map.extract_region(Coord2D::new(3, 1), 1);
		assert_eq!(extract.0, Coord2D::new(2, 0));
		assert_eq!(extract.1,
			Map::from_str(
				"000\n\
				 230\n\
				 #00"
			).unwrap()
		);
		let extract = map.extract_region(Coord2D::new(3, 1), 2);
		assert_eq!(extract.0, Coord2D::new(1, 0));
		assert_eq!(extract.1,
			Map::from_str(
				"0000#\n\
				 12302\n\
				 ##003"
			).unwrap()
		);
		let extract = map.extract_region(Coord2D::new(5, 2), 2);
		assert_eq!(extract.0, Coord2D::new(3, 0));
		assert_eq!(extract.1,
			Map::from_str(
				"00#\n\
				 302\n\
				 003"
			).unwrap()
		);
	}
}