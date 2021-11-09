use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;

use npc_engine_turn::{AgentId, Behavior, StateRef, Task};

use crate::{
    config, Barrier, Chop, Direction, Lumberjacks, Map2D, Move, Plant, Refill, State, Wait, Water,
    DIRECTIONS,
};

pub struct Lumberjack;

impl fmt::Display for Lumberjack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Lumberjacks")
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct WavefrontState {
    cost: u16,
    position: (usize, usize),
}

impl Ord for WavefrontState {
    fn cmp(&self, other: &WavefrontState) -> Ordering {
        other
            .cost
            .cmp(&self.cost)
            .then_with(|| self.position.cmp(&other.position))
    }
}

impl PartialOrd for WavefrontState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn wavefront_expansion(
    pathfindable: impl Fn(usize, usize) -> bool,
    (start_x, start_y): (usize, usize),
    radius: usize,
) -> Map2D<u16> {
    let mut heap = BinaryHeap::new();

    let extent = radius * 2 + 1;
    let mut gradient = Map2D::new_square(extent, u16::MAX);

    heap.push(WavefrontState {
        cost: 0,
        position: (start_x, start_y),
    });

    while let Some(WavefrontState {
        cost,
        position: (x, y),
    }) = heap.pop()
    {
        let gradient_value = gradient
            .get_mut(x + radius - start_x, y + radius - start_y)
            .unwrap();

        if *gradient_value <= cost {
            continue;
        }

        *gradient_value = cost;

        // X lower bound
        if x + radius >= start_x + 1 {
            if pathfindable(x - 1, y) {
                heap.push(WavefrontState {
                    cost: cost + 1,
                    position: (x - 1, y),
                });
            }
        }

        // X upper bound
        if x + 1 <= start_x + radius {
            if pathfindable(x + 1, y) {
                heap.push(WavefrontState {
                    cost: cost + 1,
                    position: (x + 1, y),
                });
            }
        }

        // Y lower bound
        if y + radius >= start_y + 1 {
            if pathfindable(x, y - 1) {
                heap.push(WavefrontState {
                    cost: cost + 1,
                    position: (x, y - 1),
                });
            }
        }

        // Y upper bound
        if y + 1 <= start_y + radius {
            if pathfindable(x, y + 1) {
                heap.push(WavefrontState {
                    cost: cost + 1,
                    position: (x, y + 1),
                });
            }
        }
    }

    gradient
}

fn wavefront_pathfind(
    gradient: &Map2D<u16>,
    (agent_x, agent_y): (usize, usize),
    (mut target_x, mut target_y): (usize, usize),
) -> Option<Vec<(usize, usize)>> {
    assert_eq!(gradient.width() % 2, 1);
    assert_eq!(gradient.width(), gradient.height());

    let radius = (gradient.width() - 1) / 2;

    let mut path = Vec::new();
    path.push((target_x, target_y));

    while target_x != agent_x || target_y != agent_y {
        let mut best_cost = u16::MAX;
        let (mut best_x, mut best_y) = (0, 0);

        let mut update_best = |x, y| {
            if let Some(&cost) = gradient.get(x + radius - agent_x, y + radius - agent_y) {
                if cost < best_cost {
                    best_cost = cost;
                    best_x = x;
                    best_y = y;
                }
            }
        };

        if target_x > 0 {
            update_best(target_x - 1, target_y);
        }
        update_best(target_x + 1, target_y);

        if target_y > 0 {
            update_best(target_x, target_y - 1);
        }
        update_best(target_x, target_y + 1);

        if best_cost != u16::MAX {
            path.push((best_x, best_y));
            target_x = best_x;
            target_y = best_y;
        } else {
            return None;
        }
    }

    path.reverse();

    Some(path)
}

impl Behavior<Lumberjacks> for Lumberjack {
    fn tasks(
        &self,
        state: StateRef<Lumberjacks>,
        agent: AgentId,
        tasks: &mut Vec<Box<dyn Task<Lumberjacks>>>,
    ) {
        if let Some((x, y)) = state.find_agent(agent) {
            if config().agents.tasks {
                // Movement
                let mut wavefront = None;

                state.points_of_interest(|target_x, target_y| {
                    if wavefront.is_none() {
                        wavefront = Some(wavefront_expansion(
                            |x: usize, y: usize| {
                                state
                                    .get_tile(x as _, y as _)
                                    .map(|tile| tile.is_pathfindable())
                                    .unwrap_or_default()
                            },
                            (x as _, y as _),
                            config().agents.snapshot_radius,
                        ));
                    }

                    let wavefront = wavefront.as_mut().unwrap();

                    if !(target_x == x && target_y == y) {
                        if let Some(path) = wavefront_pathfind(
                            &wavefront,
                            (x as _, y as _),
                            (target_x as _, target_y as _),
                        ) {
                            let task = Move {
                                path: path
                                    .windows(2)
                                    .map(|slice| {
                                        let start = slice[0];
                                        let end = slice[1];
                                        Direction::from(
                                            (start.0 as _, start.1 as _),
                                            (end.0 as _, end.1 as _),
                                        )
                                    })
                                    .collect(),
                                x: target_x as _,
                                y: target_y as _,
                            };

                            if task.valid(state, agent) {
                                tasks.push(Box::new(task));
                            }
                        }
                    }
                });
            } else {
                for &direction in DIRECTIONS {
                    let adjacent = direction.apply(x, y);

                    if state
                        .get_tile(adjacent.0, adjacent.1)
                        .map(|tile| tile.is_walkable())
                        .unwrap_or(false)
                    {
                        tasks.push(Box::new(Move {
                            path: vec![direction],
                            x: adjacent.0 as _,
                            y: adjacent.1 as _,
                        }));
                    }
                }
            }

            // Chopping
            for &direction in DIRECTIONS {
                if (Chop { direction }).valid(state, agent) {
                    tasks.push(Box::new(Chop { direction }));
                }
            }

            // Barriers
            if config().features.barriers && state.get_inventory(agent) > 0 {
                for &direction in DIRECTIONS {
                    if (Barrier { direction }).valid(state, agent) {
                        tasks.push(Box::new(Barrier { direction }));
                    }
                }
            }

            // Watering
            if config().features.watering {
                if state.get_water(agent) {
                    for &direction in DIRECTIONS {
                        if (Water { direction }.valid(state, agent)) {
                            tasks.push(Box::new(Water { direction }));
                        }
                    }
                } else {
                    if Refill.valid(state, agent) {
                        tasks.push(Box::new(Refill))
                    }
                }
            }

            // Planting
            if config().features.planting && state.get_inventory(agent) > 0 {
                for &direction in DIRECTIONS {
                    if (Plant { direction }).valid(state, agent) {
                        tasks.push(Box::new(Plant { direction }));
                    }
                }
            }

            // Waiting
            if config().features.waiting || tasks.is_empty() {
                tasks.push(Box::new(Wait));
            }
        } else {
            unreachable!()
        }
    }

    fn predicate(&self, _: StateRef<Lumberjacks>, _: AgentId) -> bool {
        true
    }
}
