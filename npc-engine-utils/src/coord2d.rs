use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign, Neg};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Coord2D {
	pub x: i32,
	pub y: i32
}
impl Coord2D {
	pub fn new(x: i32, y: i32) -> Self {
		Coord2D { x, y }
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

