/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use board::State;
use npc_engine_core::{graphviz, AgentId, MCTSConfiguration, MCTS};
use npc_engine_utils::plot_tree_in_tmp;
use regex::Regex;

use crate::{
    board::{Board, Cell, CellArray2D, CellCoord},
    domain::TicTacToe,
    player::Player,
    r#move::Move,
};

mod board;
mod domain;
mod r#move;
mod player;

enum Input {
    Coordinate((CellCoord, CellCoord)),
    Quit,
    Error,
}

fn get_input() -> Input {
    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .expect("Failed to read line");
    match input.trim() {
        "q" => Input::Quit,
        s => {
            let input_re = Regex::new(r"^([0-2])\s([0-2])$").unwrap();
            let cap = input_re.captures(s);
            let cap = cap.filter(|cap| cap.len() >= 3);
            if let Some(cap) = cap {
                let x = CellCoord::new(cap[1].parse().unwrap()).unwrap();
                let y = CellCoord::new(cap[2].parse().unwrap()).unwrap();
                Input::Coordinate((x, y))
            } else {
                println!("Input error, try again!");
                Input::Error
            }
        }
    }
}

fn run_mcts_and_return_move(
    board: u32,
    agent: AgentId,
    config: MCTSConfiguration,
    turn_to_plot: Option<u32>,
) -> Move {
    let mut mcts = MCTS::<TicTacToe>::new(board, agent, config);
    if let Some(turn) = turn_to_plot {
        if let Err(e) = plot_tree_in_tmp(&mcts, "tic-tac-toe_graphs", &format!("turn{turn:02}")) {
            println!("Cannot write search tree: {e}");
        }
    }
    let task = mcts.run().unwrap();
    task.downcast_ref::<Move>().unwrap().clone()
}

fn game_finished(state: State) -> bool {
    if state.is_full() {
        println!("Draw!");
        return true;
    }
    if let Some(winner) = state.winner() {
        match winner {
            Player::O => println!("You won!"),
            Player::X => println!("Computer won!"),
        };
        true
    } else {
        false
    }
}

fn main() {
    // These parameters control the MCTS algorithm.
    const CONFIG: MCTSConfiguration = MCTSConfiguration {
        allow_invalid_tasks: false,
        visits: 1000,
        depth: 9,
        exploration: 1.414,
        discount_hl: f32::INFINITY,
        seed: None,
        planning_task_duration: None,
    };

    // Set the depth of graph output to 6 and enable logging if specified
    // in the RUST_LOG environment variable.
    graphviz::set_graph_output_depth(6);
    env_logger::init();

    println!("Welcome to tic-tac-toe. You are player 'O', I'm player 'X'.");

    let mut board = 0;
    let mut turn = 0;
    loop {
        // Print the current board.
        println!("{}", board.description());

        // Get input.
        println!("Please enter a coordinate with 'X Y' where X,Y are 0,1,2, or 'q' to quit.");
        let (x, y) = match get_input() {
            Input::Coordinate(pair) => pair,
            Input::Quit => break,
            Input::Error => continue,
        };
        if board.get(x, y) != Cell::Empty {
            println!("The cell {x} {y} is already occupied!");
            continue;
        }

        // Set cell.
        board.set(x, y, Cell::Player(Player::O));

        // Did we win?
        if game_finished(board) {
            println!("{}", board.description());
            break;
        }

        // Run planner.
        println!("Computer is thinking...");
        const AI_AGENT: AgentId = AgentId(1);
        let ai_move = run_mcts_and_return_move(board, AI_AGENT, CONFIG, Some(turn));
        println!("Computer played {ai_move}");
        board.set(ai_move.x, ai_move.y, Cell::Player(Player::X));
        turn += 1;

        // Did computer win?
        if game_finished(board) {
            println!("{}", board.description());
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn ai_vs_ai_must_be_a_draw() {
        const CONFIG: MCTSConfiguration = MCTSConfiguration {
            allow_invalid_tasks: false,
            visits: 5000,
            depth: 9,
            exploration: 1.414,
            discount_hl: f32::INFINITY,
            planning_task_duration: None,
            seed: None,
        };
        for _ in 0..10 {
            let mut board = 0;
            loop {
                for agent in [AgentId(0), AgentId(1)] {
                    let task = run_mcts_and_return_move(board, agent, CONFIG, None);
                    board.set(task.x, task.y, Cell::Player(Player::from_agent(agent)));
                    assert_eq!(board.winner(), None);
                    if board.is_full() {
                        return;
                    }
                }
            }
        }
    }
}
