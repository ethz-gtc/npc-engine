/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use npc_engine_common::{graphviz, AgentId, MCTSConfiguration, MCTS};
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

fn main() {
    const CONFIG: MCTSConfiguration = MCTSConfiguration {
        allow_invalid_tasks: false,
        visits: 1000,
        depth: 9,
        exploration: 1.414,
        discount_hl: f32::INFINITY,
        seed: None,
        planning_task_duration: None,
    };
    graphviz::GRAPH_OUTPUT_DEPTH.store(6, std::sync::atomic::Ordering::Relaxed);
    env_logger::init();
    let mut board = 0;
    let re = Regex::new(r"^([0-2])\s([0-2])$").unwrap();
    let game_finished = |state: u32| {
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
    };
    println!("Welcome to tic-tac-toe. You are player 'O', I'm player 'X'.");
    let mut turn = 0;
    loop {
        println!("{}", board.description());

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
                let cap = cap.filter(|cap| cap.len() >= 3);
                if let Some(cap) = cap {
                    let x = CellCoord::new(cap[1].parse().unwrap()).unwrap();
                    let y = CellCoord::new(cap[2].parse().unwrap()).unwrap();
                    (x, y)
                } else {
                    println!("Input error, try again!");
                    continue;
                }
            }
        };
        if board.get(x, y) != Cell::Empty {
            println!("The cell {x} {y} is already occupied!");
            continue;
        }

        // Set cell
        board.set(x, y, Cell::Player(Player::O));

        // Did we win?
        if game_finished(board) {
            println!("{}", board.description());
            break;
        }

        // Run planner
        println!("Computer is thinking...");
        let mut mcts = MCTS::<TicTacToe>::new(board, AgentId(1), CONFIG);
        let task = mcts.run().unwrap();
        let task = task.downcast_ref::<Move>().unwrap();
        println!("Computer played {} {}", task.x, task.y);
        board.set(task.x, task.y, Cell::Player(Player::X));
        if let Err(e) = plot_tree_in_tmp(&mcts, "tic-tac-toe_graphs", &format!("turn{turn:02}")) {
            println!("Cannot write search tree: {e}");
        }
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
                    let mut mcts = MCTS::<TicTacToe>::new(board, agent, CONFIG);
                    let task = mcts.run().unwrap();
                    let task = task.downcast_ref::<Move>().unwrap();
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
