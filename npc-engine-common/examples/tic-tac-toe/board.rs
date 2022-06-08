/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use crate::player::Player;
use bounded_integer::BoundedU32;
use cached::proc_macro::cached;

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum Cell {
    Empty,
    Player(Player),
}
impl Cell {
    fn char(&self) -> char {
        match *self {
            Cell::Empty => '_',
            Cell::Player(Player::O) => 'O',
            Cell::Player(Player::X) => 'X',
        }
    }
}

pub type CellCoord = BoundedU32<0, 2>;

// SAFETY: 0, 1, 2 are in range 0..2
const C0: CellCoord = unsafe { CellCoord::new_unchecked(0) };
const C1: CellCoord = unsafe { CellCoord::new_unchecked(1) };
const C2: CellCoord = unsafe { CellCoord::new_unchecked(2) };
// TODO: once the step_trait feature is stabilized, switch to a simple range-based loop
pub const C_RANGE: [CellCoord; 3] = [C0, C1, C2];
// TODO: once const Option::unwrap() is stabilized, switch to that
pub type CoordPair = (CellCoord, CellCoord);

pub type State = u32;

pub trait CellArray2D {
    fn get(&self, x: CellCoord, y: CellCoord) -> Cell;
    fn set(&mut self, x: CellCoord, y: CellCoord, cell: Cell);
    fn description(&self) -> String;
}

impl CellArray2D for State {
    fn get(&self, x: CellCoord, y: CellCoord) -> Cell {
        let shift = y.get() * 6 + x.get() * 2;
        match (*self >> shift) & 0x3 {
            0 => Cell::Empty,
            1 => Cell::Player(Player::O),
            2 => Cell::Player(Player::X),
            _ => panic!("Invalid cell state"),
        }
    }

    fn set(&mut self, x: CellCoord, y: CellCoord, cell: Cell) {
        let pattern = match cell {
            Cell::Empty => 0,
            Cell::Player(Player::O) => 1,
            Cell::Player(Player::X) => 2,
        };
        let shift = y.get() * 6 + x.get() * 2;
        *self &= !(0b11 << shift);
        *self |= pattern << shift;
    }

    fn description(&self) -> String {
        let mut s = String::new();
        for y in C_RANGE {
            for x in C_RANGE {
                s.push(self.get(x, y).char());
                if x != C2 {
                    s.push(' ');
                }
            }
            if y != C2 {
                s.push('\n');
            }
        }
        s
    }
}

pub trait Board {
    fn is_line_all_of(&self, player: Player, line: &[CoordPair]) -> bool;
    fn is_full(&self) -> bool;
    fn winner(&self) -> Option<Player>;
}

impl Board for State {
    fn is_line_all_of(&self, player: Player, line: &[CoordPair]) -> bool {
        for (x, y) in line {
            if CellArray2D::get(self, *x, *y) != Cell::Player(player) {
                return false;
            }
        }
        true
    }

    fn is_full(&self) -> bool {
        for x in C_RANGE {
            for y in C_RANGE {
                if self.get(x, y) == Cell::Empty {
                    return false;
                }
            }
        }
        true
    }

    fn winner(&self) -> Option<Player> {
        cached_winner(*self)
    }
}

#[cached(size = 19683)] // there are 3^9 possible states
fn cached_winner(state: State) -> Option<Player> {
    const LINES: [[CoordPair; 3]; 8] = [
        // diagonals
        [(C0, C0), (C1, C1), (C2, C2)],
        [(C0, C2), (C1, C1), (C2, C0)],
        // horizontals
        [(C0, C0), (C1, C0), (C2, C0)],
        [(C0, C1), (C1, C1), (C2, C1)],
        [(C0, C2), (C1, C2), (C2, C2)],
        // verticals
        [(C0, C0), (C0, C1), (C0, C2)],
        [(C1, C0), (C1, C1), (C1, C2)],
        [(C2, C0), (C2, C1), (C2, C2)],
    ];
    for line in &LINES {
        for player in [Player::O, Player::X] {
            if state.is_line_all_of(player, line) {
                return Some(player);
            }
        }
    }
    None
}

pub type Diff = Option<State>; // if Some, use this diff, otherwise use initial state

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_array_2d() {
        let mut state: State = 0;
        CellArray2D::set(&mut state, C0, C0, Cell::Empty);
        assert_eq!(state, 0b00);
        CellArray2D::set(&mut state, C0, C0, Cell::Player(Player::O));
        assert_eq!(state, 0b01);
        CellArray2D::set(&mut state, C0, C0, Cell::Player(Player::X));
        assert_eq!(state, 0b10);
        CellArray2D::set(&mut state, C2, C0, Cell::Player(Player::X));
        assert_eq!(state, 0b100010);
        CellArray2D::set(&mut state, C0, C2, Cell::Player(Player::O));
        assert_eq!(state, 0b000001_000000_100010);
        CellArray2D::set(&mut state, C2, C2, Cell::Player(Player::X));
        assert_eq!(state, 0b100001_000000_100010);
        CellArray2D::set(&mut state, C0, C0, Cell::Empty);
        assert_eq!(state, 0b100001_000000_100000);
        CellArray2D::set(&mut state, C1, C1, Cell::Player(Player::X));
        assert_eq!(state, 0b100001_001000_100000);
    }

    #[test]
    fn winner() {
        let winner = |state: State| state.winner();
        assert_eq!(winner(0), None);
        assert_eq!(winner(0b000000_000000_010101), Some(Player::O));
        assert_eq!(winner(0b000000_010101_000000), Some(Player::O));
        assert_eq!(winner(0b010101_000000_000000), Some(Player::O));
        assert_eq!(winner(0b100000_100000_100000), Some(Player::X));
        assert_eq!(winner(0b001000_001000_001000), Some(Player::X));
        assert_eq!(winner(0b000010_000010_000010), Some(Player::X));
        assert_eq!(winner(0b100001_001000_010010), Some(Player::X));
        assert_eq!(winner(0b100001_000100_010010), Some(Player::O));
        assert_eq!(winner(0b100010_000100_010010), None);
    }
}
