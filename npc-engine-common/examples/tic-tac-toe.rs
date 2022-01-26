use std::{collections::BTreeSet, fmt};
use bounded_integer::BoundedU32;
use cached::proc_macro::cached;

use npc_engine_common::{Domain, Behavior, StateDiffRef, AgentId, AgentValue, Task, StateDiffRefMut, impl_task_boxed_methods, MCTS, MCTSConfiguration};
use regex::Regex;

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
enum Player {
	O,
	X
}
impl Player {
	fn from_agent(agent: AgentId) -> Self {
		match agent {
			AgentId(0) => Player::O,
			AgentId(1) => Player::X,
			AgentId(id) => panic!("Invalid AgentId {id}")
		}
	}
	fn to_agent(self) -> AgentId {
		match self {
			Player::O => AgentId(0),
			Player::X => AgentId(1)
		}
	}
}
impl fmt::Display for Player {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Player::O => write!(f, "O"),
			Player::X => write!(f, "X")
		}
	}
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
enum Cell {
	Empty,
	Player(Player)
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

type CellCoord = BoundedU32<0, 2>;
trait CellArray2D {
	fn get(&self, x: CellCoord, y: CellCoord) -> Cell;
	fn set(&mut self, x: CellCoord, y: CellCoord, cell: Cell);
	fn print(&self);
}
type State = u32;
impl CellArray2D for State {
	fn get(&self, x: CellCoord, y: CellCoord) -> Cell {
		let shift = y.get() * 6 + x.get() * 2;
		match (*self >> shift) & 0x3 {
			0 => Cell::Empty,
			1 => Cell::Player(Player::O),
			2 => Cell::Player(Player::X),
			_ => panic!("Invalid cell state")
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
	fn print(&self) {
		for y in C_RANGE {
			for x in C_RANGE {
				print!("{} ", self.get(x, y).char());
			}
			println!();
		}
	}
}

type Diff = Option<State>; // if Some, use this diff, otherwise use initial state
// TODO: once const Option::unwrap() is stabilized, switch to that
// SAFETY: 0, 1, 2 are in range 0..2
const C0: CellCoord = unsafe { CellCoord::new_unchecked(0) };
const C1: CellCoord = unsafe { CellCoord::new_unchecked(1) };
const C2: CellCoord = unsafe { CellCoord::new_unchecked(2) };
// TODO: once the step_trait feature is stabilized, switch to a simple range-based loop
const C_RANGE: [CellCoord; 3] = [C0, C1, C2];
type CoordPair = (CellCoord, CellCoord);

fn is_line_all_of(state: State, player: Player, line: &[CoordPair]) -> bool {
	for (x, y) in line {
		if CellArray2D::get(&state, *x, *y) != Cell::Player(player) {
			return false;
		}
	}
	true
}

#[cached(size=19683)] // there are 3^9 possible states
fn winner(state: State) -> Option<Player> {
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
			if is_line_all_of(state, player, line) {
				return Some(player)
			}
		}
	}
	None
}

fn board_full(state: State) -> bool {
	for x in C_RANGE {
		for y in C_RANGE {
			if state.get(x, y) == Cell::Empty {
				return false;
			}
		}
	}
	true
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct Move {
	x: CellCoord,
	y: CellCoord,
}

struct TicTacToe;

impl Task<TicTacToe> for Move {
	fn execute(
        &self,
        _tick: u64,
        state_diff: StateDiffRefMut<TicTacToe>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<TicTacToe>>> {
		let diff = if let Some(diff) = state_diff.diff {
			diff
		} else {
			*state_diff.diff = Some(0);
			&mut *state_diff.diff.as_mut().unwrap()
		};
		diff.set(self.x, self.y, Cell::Player(Player::from_agent(agent)));
		None
	}

	fn display_action(&self) -> <TicTacToe as Domain>::DisplayAction {
        Some(self.clone())
    }

	fn is_valid(&self, _tick: u64, state_diff: StateDiffRef<TicTacToe>, _agent: AgentId) -> bool {
		let state = *state_diff.initial_state | state_diff.diff.unwrap_or(0);
		state.get(self.x, self.y) == Cell::Empty
    }

    impl_task_boxed_methods!(TicTacToe);
}

impl fmt::Debug for Move {
	fn fmt(&self, f: &'_ mut fmt::Formatter) -> fmt::Result {
		f.debug_struct("Move")
		.field("x", &self.x.get())
		.field("y", &self.y.get())
		.finish()
    }
}

struct MoveBehavior;
impl Behavior<TicTacToe> for MoveBehavior {
	fn add_own_tasks(
		&self,
		tick: u64,
		state_diff: StateDiffRef<TicTacToe>,
		agent: AgentId,
		tasks: &mut Vec<Box<dyn Task<TicTacToe>>>,
	) {
		for x in C_RANGE {
			for y in C_RANGE {
				let task = Move { x, y };
				if task.is_valid(tick, state_diff, agent) {
					tasks.push(Box::new(task));
				}
			}
		}
	}

	fn is_valid(&self, _tick: u64, _state_diff: StateDiffRef<TicTacToe>, _agent: AgentId) -> bool {
		true
	}
}

// TODO: once const NotNan::new() is stabilized, switch to that
// SAFETY: 0.0, 1.0, -1.0 are not NaN
const VALUE_UNDECIDED: AgentValue = unsafe { AgentValue::new_unchecked(0.) };
const VALUE_WIN: AgentValue = unsafe { AgentValue::new_unchecked(1.) };
const VALUE_LOOSE: AgentValue = unsafe { AgentValue::new_unchecked(-1.) };

impl Domain for TicTacToe {
	type State = State;
	type Diff = Diff;
	type DisplayAction = Option<Move>; // Option, so that the idle placeholder action is None

	fn list_behaviors() -> &'static [&'static dyn Behavior<Self>] {
		&[&MoveBehavior]
	}

	fn get_current_value(_tick: u64, state_diff: StateDiffRef<Self>, agent: AgentId) -> AgentValue {
		let state = *state_diff.initial_state | state_diff.diff.unwrap_or(0);
		match winner(state) {
			None => VALUE_UNDECIDED,
			Some(player) => if player.to_agent() == agent { VALUE_WIN } else { VALUE_LOOSE },
		}
	}

	fn update_visible_agents(_tick: u64, _state_diff: StateDiffRef<Self>, _agent: AgentId, agents: &mut BTreeSet<AgentId>) {
		agents.insert(AgentId(0));
		agents.insert(AgentId(1));
	}
}


fn main() {
	const CONFIG: MCTSConfiguration = MCTSConfiguration {
		visits: 5000,
		depth: 9,
		exploration: 1.414,
		discount_hl: f32::INFINITY,
		seed: None
	};
	env_logger::init();
	let mut state = 0;
	let re = Regex::new(r"^([0-2])\s([0-2])$").unwrap();
	let game_finished = |state: u32| {
		if board_full(state) {
			println!("Draw!");
			return true;
		}
		if let Some(winner) = winner(state) {
			match winner {
				Player::O => println!("You won!"),
				Player::X => println!("Computer won!")
			};
			true
		} else {
			false
		}
	};
	println!("Welcome to tic-tac-toe. You are player 'O', I'm player 'X'.");
	loop {
		state.print();

		// Get input
		println!("Please enter a coordinate with 'X Y' where X,Y are 0,1,2, or 'q' to quit.");
		let mut input = String::new();
		std::io::stdin()
			.read_line(&mut input)
			.expect("Failed to read line");
		// println!("Input string: '{}'", input.trim());
		let (x, y) = match input.trim() {
			"q" => break,
			s => {
				let cap = re.captures(s);
				if cap.is_none() || cap.as_ref().unwrap().len() < 3 {
					println!("Input error, try again!");
					continue;
				} else {
					let cap = cap.unwrap();
					let x = CellCoord::new(cap[1].parse().unwrap()).unwrap();
					let y = CellCoord::new(cap[2].parse().unwrap()).unwrap();
					(x, y)
				}
			}
		};
		if state.get(x, y) != Cell::Empty {
			println!("The cell {x} {y} is already occupied!");
			continue;
		}
		
		// Set cell
		state.set(x, y, Cell::Player(Player::O));

		// Did we win?
		if game_finished(state) {
			break;
		}

		// Run planner
		println!("Computer is thinking...");
		let mut mcts = MCTS::<TicTacToe>::new(
			state,
			AgentId(1),
			CONFIG
		);
		let task = mcts.run();
		let task = task.downcast_ref::<Move>().unwrap();
		println!("Computer played {} {}", task.x, task.y);
		state.set(task.x, task.y, Cell::Player(Player::X));

		// Did computer win?
		if game_finished(state) {
			break;
		}
	}
}

#[cfg(test)]
mod tests {
    use crate::*;

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
		use crate::winner;
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

	#[test]
	fn ai_vs_ai_must_be_a_draw() {
		use crate::winner;
		const CONFIG: MCTSConfiguration = MCTSConfiguration {
			visits: 5000,
			depth: 9,
			exploration: 1.414,
			discount_hl: f32::INFINITY,
			seed: None
		};
		let mut state = 0;
		loop {
			for agent in [AgentId(0), AgentId(1)] {
				let mut mcts = MCTS::<TicTacToe>::new(
					state,
					agent,
					CONFIG
				);
				let task = mcts.run();
				let task = task.downcast_ref::<Move>().unwrap();
				state.set(task.x, task.y, Cell::Player(Player::from_agent(agent)));
				assert_eq!(winner(state), None);
				if board_full(state) {
					return;
				}
			}
		}
	}
}