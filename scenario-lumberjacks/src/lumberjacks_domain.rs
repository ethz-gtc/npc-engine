use std::collections::BTreeSet;

use npc_engine_common::{Domain, Behavior, AgentId, StateDiffRef, AgentValue};
use npc_engine_utils::GlobalDomain;

use crate::{WorldLocalState, WorldDiff, Human, Lumberjack, config, Action, WorldGlobalState, WorldState, TileMapSnapshot, Tile, InventorySnapshot, AgentInventory};

pub struct Lumberjacks;

impl Domain for Lumberjacks {
    type State = WorldLocalState;
    type Diff = WorldDiff;
    type DisplayAction = Action;

    fn list_behaviors() -> &'static [&'static dyn Behavior<Self>] {
        &[&Human, &Lumberjack]
    }

    fn get_current_value(state_diff: StateDiffRef<Self>, agent: AgentId) -> AgentValue {
        let value = if let Some((_, f)) = config().agents.behaviors.get(&(agent.0 as usize)) {
            f(state_diff, agent)
        } else {
            state_diff.get_inventory(agent) as f32
        };
        AgentValue::new(value).unwrap()
    }

    fn update_visible_agents(state_diff: StateDiffRef<Self>, agent: AgentId, agents: &mut BTreeSet<AgentId>) {
        if let Some((x, y)) = state_diff.find_agent(agent) {
            if config().agents.plan_others {
                agents.extend(
                    state_diff
                        .find_nearby_agents(x, y, config().agents.horizon_radius)
                        .into_iter(),
                );
            } else {
                agents.insert(agent);
            }
        } else {
            unreachable!("{:?}", state_diff);
        }
    }
}

impl GlobalDomain for Lumberjacks {
    type GlobalState = WorldGlobalState;

    fn derive_local_state(state: &Self::GlobalState, agent: AgentId) -> Self::State {
        let (x, y) = state.find_agent(agent).unwrap();

        let top = y - config().agents.snapshot_radius as isize;
        let left = x - config().agents.snapshot_radius as isize;

        let mut map = TileMapSnapshot {
            top,
            left,
            tiles: (0..(config().agents.snapshot_radius * 2 + 1))
                .map(|_| {
                    (0..(config().agents.snapshot_radius * 2 + 1))
                        .map(|_| Tile::Empty)
                        .collect::<Vec<_>>()
                        .into_boxed_slice()
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        };

        for y in 0..(config().agents.snapshot_radius * 2 + 1) {
            for x in 0..(config().agents.snapshot_radius * 2 + 1) {
                let src_y = top + y as isize;
                let src_x = left + x as isize;

                if (src_y >= 0 && src_y < state.map.height as isize)
                    && (src_x >= 0 && src_x < state.map.width as isize)
                {
                    map.tiles[y as usize][x as usize] =
                        state.map.tiles[(top + y as isize) as usize][(left + x as isize) as usize];
                } else {
                    map.tiles[y as usize][x as usize] = Tile::Impassable;
                }
            }
        }

        WorldLocalState {
            inventory: InventorySnapshot(state.inventory.0.clone()),
            map,
        }
    }

    fn apply(state: &mut Self::GlobalState, snapshot: &Self::State, diff: &Self::Diff) {
        for (agent, AgentInventory { wood, water }) in &diff.inventory.0 {
            if let Some(inventory) = state.inventory.0.get_mut(agent) {
                inventory.wood += *wood;
                inventory.water = *water;
            }
        }

        for ((x, y), tile) in &diff.map.tiles {
            let dest_y = y + snapshot.map.top;
            let dest_x = x + snapshot.map.left;
            if (dest_y >= 0 && dest_y < state.map.height as isize)
                && (dest_x >= 0 && dest_x < state.map.width as isize)
            {
                state.map.tiles[dest_y as usize][dest_x as usize] = *tile;
            }
        }
    }

    /*
    // Note: this was used for fuzzy node reuse, but that is too complex for now
    fn compatible(snapshot: &Self::Snapshot, other: &Self::Snapshot, agent: AgentId) -> bool {
        let (y, x) = snapshot
            .map
            .tiles
            .iter()
            .enumerate()
            .find_map(|(row, cols)| {
                cols.iter().enumerate().find_map(|(col, tile)| {
                    if *tile == Tile::Agent(agent) {
                        Some((row, col))
                    } else {
                        None
                    }
                })
            })
            .unwrap();

        let (other_y, other_x) = other
            .map
            .tiles
            .iter()
            .enumerate()
            .find_map(|(row, cols)| {
                cols.iter().enumerate().find_map(|(col, tile)| {
                    if *tile == Tile::Agent(agent) {
                        Some((row, col))
                    } else {
                        None
                    }
                })
            })
            .unwrap();

        (-(config().agents.horizon_radius as isize)..=config().agents.horizon_radius as isize).all(
            |row| {
                (-(config().agents.horizon_radius as isize)
                    ..config().agents.horizon_radius as isize)
                    .all(|col| {
                        let row0 = y as isize + row;
                        let col0 = x as isize + col;
                        let row1 = other_y as isize + row;
                        let col1 = other_x as isize + col;
                        let size = config().agents.snapshot_radius as isize * 2 + 1;

                        if row0 >= 0
                            && row0 < size
                            && col0 >= 0
                            && col0 < size
                            && row1 >= 0
                            && row1 < size
                            && col1 >= 0
                            && col1 < size
                        {
                            snapshot.map.tiles[(y as isize + row) as usize]
                                [(x as isize + col) as usize]
                                == other.map.tiles[(other_y as isize + row) as usize]
                                    [(other_x as isize + col) as usize]
                        } else {
                            true
                        }
                    })
            },
        )
    }
    */
}