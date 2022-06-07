/* 
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign, Neg};

use rand::Rng;

/// A 2-D integer coordinate type.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Coord2D {
	pub x: i32,
	pub y: i32
}
impl Coord2D {
	pub const fn new(x: i32, y: i32) -> Self {
		Coord2D { x, y }
	}
    /// Returns the absolute value, component by component.
    pub const fn abs(&self) -> Self {
        Self::new(self.x.abs(), self.y.abs())
    }
    /// Returns the absolute value of the difference to other, component by component.
    pub fn abs_diff(&self, other: &Coord2D) -> Self {
        (*self - *other).abs()
    }
    /// Returns the value of the smallest dimension of the absolute value of the difference to other.
    pub fn shortest_dim_dist(&self, other: &Coord2D) -> i32 {
        let diff = self.abs_diff(other);
        diff.x.min(diff.y)
    }
    /// Returns the value of the largest dimension of the absolute value of the difference to other.
    pub fn largest_dim_dist(&self, other: &Coord2D) -> i32 {
        let diff = self.abs_diff(other);
        diff.x.max(diff.y)
    }
    /// Returns the maximum value of self and other, component by component.
    pub fn max_per_comp(&self, other: Coord2D) -> Self {
        Self::new(
            self.x.max(other.x),
            self.y.max(other.y),
        )
    }
    /// Returns the minimum value of self and other, component by component.
    pub fn min_per_comp(&self, other: Coord2D) -> Self {
        Self::new(
            self.x.min(other.x),
            self.y.min(other.y),
        )
    }
    /// Generates a random value between 0 and range.
    pub fn rand_uniform(range: Coord2D) -> Self {
        let mut rng = rand::thread_rng();
        let x = rng.gen_range(0..range.x);
        let y = rng.gen_range(0..range.y);
        Self::new(x, y)
    }
    /// Manhattan distance between self and other.
    pub fn manhattan_dist(&self, other: Coord2D) -> i32 {
        let diff = self.abs_diff(&other);
        diff.x + diff.y
    }
}

impl std::fmt::Display for Coord2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}, {}]", self.x, self.y)
    }
}

impl Ord for Coord2D {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
		self.y.cmp(&other.y).then(self.x.cmp(&other.x))
    }
}
impl PartialOrd for Coord2D {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
		Some(self.cmp(other))
    }
}

impl Add for Coord2D{
    type Output = Coord2D;

    fn add(self, rhs: Self) -> Self::Output{
        Coord2D::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl Sub for Coord2D{
    type Output = Coord2D;

    fn sub(self, rhs: Self) -> Self::Output{
        Coord2D::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl AddAssign for Coord2D{
    fn add_assign(&mut self, rhs: Self){
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl SubAssign for Coord2D{
    fn sub_assign(&mut self, rhs: Self){
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl MulAssign<i32> for Coord2D{
    fn mul_assign(&mut self, rhs: i32){
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl DivAssign<i32> for Coord2D{
    fn div_assign(&mut self, rhs: i32){
        self.x /= rhs;
        self.y /= rhs;
    }
}

impl Mul<i32> for Coord2D{
    type Output = Coord2D;

    fn mul(self, rhs: i32) -> Self::Output{
        Coord2D::new(self.x * rhs, self.y * rhs)
    }
}

impl Div<i32> for Coord2D{
    type Output = Coord2D;

    fn div(self, rhs: i32) -> Self::Output{
        Coord2D::new(self.x / rhs, self.y / rhs)
    }
}

impl Neg for Coord2D{
    type Output = Coord2D;

    fn neg(self) -> Self::Output{
        Coord2D::new(-self.x, -self.y)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
	fn cmp() {
		assert_eq!(Coord2D::new(0, 0), Coord2D::new(0, 0));
		assert!(Coord2D::new(0, 0) < Coord2D::new(0, 1));
		assert!(Coord2D::new(0, 0) < Coord2D::new(1, 0));
		assert!(Coord2D::new(1, 0) < Coord2D::new(0, 1));
	}

    #[test]
    fn abs_diff() {
        assert_eq!(Coord2D::new(3, 2).abs_diff(&Coord2D::new(0, 3)), Coord2D::new(3, 1));
    }

    #[test]
    fn dim_dist() {
        assert_eq!(Coord2D::new(3, 2).shortest_dim_dist(&Coord2D::new(0, 1)), 1);
        assert_eq!(Coord2D::new(3, 2).largest_dim_dist(&Coord2D::new(0, 1)), 3);
    }

    #[test]
    fn manhattan_dist() {
        assert_eq!(Coord2D::new(0, 0).manhattan_dist(Coord2D::new(0, 2)), 2);
        assert_eq!(Coord2D::new(0, 0).manhattan_dist(Coord2D::new(2, 0)), 2);
        assert_eq!(Coord2D::new(0, 0).manhattan_dist(Coord2D::new(0, -2)), 2);
        assert_eq!(Coord2D::new(0, 0).manhattan_dist(Coord2D::new(-2, 0)), 2);
        assert_eq!(Coord2D::new(3, 2).manhattan_dist(Coord2D::new(0, 1)), 4);
    }
}