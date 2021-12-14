use std::collections::{BTreeMap, BTreeSet};
use std::usize;

use ggez::graphics;
use ggez::graphics::Image;
use ggez::Context;
use npc_engine_turn::{AgentId, StateDiffRef, StateDiffRefMut};
use npc_engine_utils::GlobalDomain;
use serde::Serialize;

use crate::{Action, AgentInventory, DIRECTIONS, Inventory, InventoryDiff, InventorySnapshot, Lumberjacks, SPRITE_SIZE, Tile, TileMap, TileMapDiff, TileMapSnapshot, config};

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "kebab-case")]
pub struct WorldState {
    pub actions: BTreeMap<AgentId, Action>,
    pub inventory: Inventory,
    pub map: TileMap,
}

impl WorldState {
    pub fn draw(&self, ctx: &mut Context, assets: &BTreeMap<String, Image>) {
        let screen = graphics::screen_coordinates(ctx);

        if config().display.inventory {
            self.inventory.draw(ctx, assets);
        }
        self.with_map_coordinates(ctx, |ctx| {
            self.map.draw(ctx, assets, &self.actions);
        });
        graphics::set_screen_coordinates(ctx, screen).unwrap();
    }

    // Draw to non-padding area
    pub fn with_map_coordinates(&self, ctx: &mut Context, f: impl Fn(&mut Context)) {
        let screen = graphics::screen_coordinates(ctx);
        graphics::set_screen_coordinates(
            ctx,
            graphics::Rect {
                x: screen.x - config().display.padding.0 as f32 * SPRITE_SIZE,
                y: screen.y - config().display.padding.1 as f32 * SPRITE_SIZE,
                w: screen.w,
                h: screen.h,
            },
        )
        .unwrap();
        f(ctx);
        graphics::set_screen_coordinates(ctx, screen).unwrap();
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WorldSnapshot {
    pub inventory: InventorySnapshot,
    pub map: TileMapSnapshot,
}

#[derive(Clone, Debug, Default, Hash, PartialEq, Eq)]
pub struct WorldDiff {
    pub ctr: usize,
    pub inventory: InventoryDiff,
    pub map: TileMapDiff,
}

impl WorldDiff {
    pub fn diff_size(&self) -> usize {
        self.inventory.diff_size() + self.map.diff_size()
    }
}

pub trait State {
    fn get_tile(&self, x: isize, y: isize) -> Option<Tile>;
    fn find_agent(&self, agent: AgentId) -> Option<(isize, isize)>;
    fn get_inventory(&self, agent: AgentId) -> usize;
    fn get_total_inventory(&self) -> usize;
    fn get_water(&self, agent: AgentId) -> bool;
    fn trees(&self) -> BTreeSet<(isize, isize)>;
    fn points_of_interest(&self, f: impl FnMut(isize, isize));
    fn find_nearby_agents(&self, x: isize, y: isize, radius: usize) -> Vec<AgentId>;
}

pub trait StateMut {
    fn increment_time(&mut self);
    fn set_tile(&mut self, x: isize, y: isize, tile: Tile);
    fn get_tile_ref_mut(&mut self, x: isize, y: isize) -> Option<&mut Tile>;
    fn increment_inventory(&mut self, agent: AgentId);
    fn decrement_inventory(&mut self, agent: AgentId);
    fn set_water(&mut self, agent: AgentId, value: bool);
}

pub(crate) enum GlobalStateRef<'a, D: GlobalDomain> {
    State(&'a D::GlobalState),
    Snapshot(StateDiffRef<'a, D>),
}

impl State for GlobalStateRef<'_, Lumberjacks> {
    fn get_tile(&self, x: isize, y: isize) -> Option<Tile> {
        match self {
            GlobalStateRef::State(state) => {
                if x >= 0 && x < state.map.width as isize && y >= 0 && y < state.map.height as isize
                {
                    Some(state.map.tiles[y as usize][x as usize])
                } else {
                    None
                }
            }
            GlobalStateRef::Snapshot(StateDiffRef { initial_state: snapshot, diff }) => {
                let (x, y) = (x - snapshot.map.left, y - snapshot.map.top);

                if x >= 0
                    && x < (config().agents.snapshot_radius * 2 + 1) as isize
                    && y >= 0
                    && y < (config().agents.snapshot_radius * 2 + 1) as isize
                {
                    if let Some(tile) = diff.map.tiles.get(&(x, y)) {
                        Some(*tile)
                    } else {
                        Some(snapshot.map.tiles[y as usize][x as usize])
                    }
                } else {
                    None
                }
            }
        }
    }

    fn trees(&self) -> BTreeSet<(isize, isize)> {
        match self {
            GlobalStateRef::State(state) => {
                let mut set = BTreeSet::new();
                for y in 0..state.map.height {
                    for x in 0..state.map.width {
                        if matches!(state.map.tiles[y][x], Tile::Tree(_)) {
                            set.insert((x as _, y as _));
                        }
                    }
                }
                set
            }
            GlobalStateRef::Snapshot(StateDiffRef { initial_state: snapshot, diff: _ }) => {
                let mut set = BTreeSet::new();
                let top = snapshot.map.top;
                let left = snapshot.map.left;
                for y in top..(top + config().agents.snapshot_radius as isize * 2 + 1) {
                    for x in left..(left + config().agents.snapshot_radius as isize * 2 + 1) {
                        if matches!(self.get_tile(x, y), Some(Tile::Tree(_))) {
                            set.insert((x, y));
                        }
                    }
                }
                set
            }
        }
    }

    fn points_of_interest(&self, mut f: impl FnMut(isize, isize)) {
        let (start_x, end_x, start_y, end_y) = match self {
            GlobalStateRef::State(state) => (
                0isize,
                state.map.width as isize,
                0isize,
                state.map.height as isize,
            ),
            GlobalStateRef::Snapshot(StateDiffRef { initial_state: snapshot, diff: _ }) => {
                let extent = config().agents.snapshot_radius as isize * 2 + 1;

                (
                    snapshot.map.left,
                    snapshot.map.left + extent,
                    snapshot.map.top,
                    snapshot.map.top + extent,
                )
            }
        };

        for y in start_y..end_y {
            for x in start_x..end_x {
                let is_poi = self
                    .get_tile(x, y)
                    .map(|tile| tile.is_pathfindable())
                    .unwrap_or(false)
                    && DIRECTIONS.into_iter().any(|direction| {
                        let (x, y) = direction.apply(x, y);
                        self.get_tile(x, y)
                            .map(|tile| tile.is_point_of_interest())
                            .unwrap_or(false)
                    });
                if is_poi {
                    f(x, y);
                }
            }
        }
    }

    fn find_agent(&self, agent: AgentId) -> Option<(isize, isize)> {
        match self {
            GlobalStateRef::State(state) => {
                for y in 0..state.map.height {
                    for x in 0..state.map.width {
                        if state.map.tiles[y][x] == Tile::Agent(agent) {
                            return Some((x as isize, y as isize));
                        }
                    }
                }

                None
            }
            GlobalStateRef::Snapshot(StateDiffRef { initial_state: snapshot, diff }) => {
                if let Some(pos) = diff.map.agents_pos.get(&agent) {
                    return Some(*pos);
                }

                for y in 0..(config().agents.snapshot_radius * 2 + 1) as isize {
                    for x in 0..(config().agents.snapshot_radius * 2 + 1) as isize {
                        if let Some(tile) = diff.map.tiles.get(&(x, y)) {
                            if *tile == Tile::Agent(agent) {
                                return Some((x + snapshot.map.left, y + snapshot.map.top));
                            }
                        } else if snapshot.map.tiles[y as usize][x as usize] == Tile::Agent(agent) {
                            return Some((x + snapshot.map.left, y + snapshot.map.top));
                        }
                    }
                }

                None
            }
        }
    }

    fn get_inventory(&self, agent: AgentId) -> usize {
        match self {
            GlobalStateRef::State(state) => state.inventory.0.get(&agent).unwrap().wood as usize,
            GlobalStateRef::Snapshot(StateDiffRef { initial_state: snapshot, diff }) => {
                (snapshot.inventory.0.get(&agent).unwrap().wood
                    + diff
                        .inventory
                        .0
                        .get(&agent)
                        .map(|inv| inv.wood)
                        .unwrap_or(0)) as usize
            }
        }
    }

    fn get_total_inventory(&self) -> usize {
        match self {
            GlobalStateRef::State(state) => state
                .inventory
                .0
                .values()
                .map(|inv| inv.wood)
                .sum::<isize>() as usize,
            GlobalStateRef::Snapshot(StateDiffRef { initial_state: snapshot, diff }) => {
                (snapshot
                    .inventory
                    .0
                    .values()
                    .map(|inv| inv.wood)
                    .sum::<isize>()
                    + diff.inventory.0.values().map(|inv| inv.wood).sum::<isize>())
                    as usize
            }
        }
    }

    fn get_water(&self, agent: AgentId) -> bool {
        match self {
            GlobalStateRef::State(state) => state
                .inventory
                .0
                .get(&agent)
                .map(|inv| inv.water)
                .unwrap_or_default(),
            GlobalStateRef::Snapshot(StateDiffRef { initial_state: snapshot, diff }) => diff
                .inventory
                .0
                .get(&agent)
                .map(|inv| inv.water)
                .unwrap_or_else(|| {
                    snapshot
                        .inventory
                        .0
                        .get(&agent)
                        .map(|inv| inv.water)
                        .unwrap_or_default()
                }),
        }
    }

    fn find_nearby_agents(&self, x: isize, y: isize, radius: usize) -> Vec<AgentId> {
        let mut vec = Vec::new();

        for y in (y - radius as isize)..=(y + radius as isize) {
            for x in (x - radius as isize)..=(x + radius as isize) {
                if let Some(Tile::Agent(agent)) = self.get_tile(x, y) {
                    vec.push(agent);
                }
            }
        }

        vec
    }
}

impl StateMut for StateDiffRefMut<'_, Lumberjacks> {
    fn increment_time(&mut self) {
        self.diff.ctr += 1;
    }

    fn set_tile(&mut self, x: isize, y: isize, tile: Tile) {
        if let Tile::Agent(agent) = tile {
            self.diff.map.agents_pos.insert(agent, (x, y));
        }

        let snapshot = self.initial_state;
        let (x, y) = (x - snapshot.map.left, y - snapshot.map.top);

        if x >= 0
            && x < (config().agents.snapshot_radius * 2 + 1) as isize
            && y >= 0
            && y < (config().agents.snapshot_radius * 2 + 1) as isize
        {
            if snapshot.map.tiles[y as usize][x as usize] == tile {
                self.diff.map.tiles.remove(&(x, y));
            } else {
                self.diff.map.tiles.insert((x, y), tile);
            }
        }
    }

    fn get_tile_ref_mut(&mut self, x: isize, y: isize) -> Option<&mut Tile> {
        let snapshot = self.initial_state;
        let (x, y) = (x - snapshot.map.left, y - snapshot.map.top);

        if x >= 0
            && x < (config().agents.snapshot_radius * 2 + 1) as isize
            && y >= 0
            && y < (config().agents.snapshot_radius * 2 + 1) as isize
        {
            Some(
                self.diff.map
                    .tiles
                    .entry((x, y))
                    .or_insert_with(|| snapshot.map.tiles[y as usize][x as usize]),
            )
        } else {
            None
        }
    }

    fn increment_inventory(&mut self, agent: AgentId) {
        let snapshot = self.initial_state;
        // this is cumbersome because the diff has a real diff for the tree (+= diff.tree)
        // but for the water it is an override (= diff.water), so we need to fetch the
        // water from the snapshot when we create a new inventory diff
        self.diff.inventory.0.entry(agent).or_insert_with(
            || AgentInventory {
                wood: 0,
                water: snapshot.inventory.0
                    .get(&agent)
                    .map(|inv| inv.water)
                    .unwrap_or_default()
            }
        ).wood += 1;
    }

    fn decrement_inventory(&mut self, agent: AgentId) {
        let snapshot = self.initial_state;
        // this is cumbersome because the diff has a real diff for the tree (+= diff.tree)
        // but for the water it is an override (= diff.water), so we need to fetch the
        // water from the snapshot when we create a new inventory diff
        self.diff.inventory.0.entry(agent).or_insert_with(
            || AgentInventory {
                wood: 0,
                water: snapshot.inventory.0
                    .get(&agent)
                    .map(|inv| inv.water)
                    .unwrap_or_default()
            }
        ).wood -= 1;
    }

    fn set_water(&mut self, agent: AgentId, value: bool) {
        self.diff.inventory.0.entry(agent).or_default().water = value;
    }
}
