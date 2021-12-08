use std::collections::{BTreeMap, BTreeSet};
use std::usize;
use std::ops::Deref;
use std::mem;

use ggez::graphics;
use ggez::graphics::Image;
use ggez::Context;
use npc_engine_turn::{AgentId, SnapshotDiffRef, SnapshotDiffRefMut, Domain};
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

pub(crate) enum StateRef<'a, D: Domain> {
    State(&'a D::State),
    Snapshot(SnapshotDiffRef<'a, D>),
}

pub(crate) enum StateRefMut<'a, D: Domain> {
    State(&'a mut D::State),
    Snapshot(SnapshotDiffRefMut<'a, D>),
}

impl<'a, D: Domain> Deref for StateRefMut<'a, D> {
    type Target = StateRef<'a, D>;

    fn deref(&self) -> &Self::Target {
        // Safety: StateRef and StateRefMut have the same memory layout
        // and casting from mutable to immutable is always safe
        unsafe { mem::transmute(self) }
    }
}

impl State for StateRef<'_, Lumberjacks> {
    fn get_tile(&self, x: isize, y: isize) -> Option<Tile> {
        match self {
            StateRef::State(state) => {
                if x >= 0 && x < state.map.width as isize && y >= 0 && y < state.map.height as isize
                {
                    Some(state.map.tiles[y as usize][x as usize])
                } else {
                    None
                }
            }
            StateRef::Snapshot(SnapshotDiffRef { snapshot, diff }) => {
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
            StateRef::State(state) => {
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
            StateRef::Snapshot(SnapshotDiffRef { snapshot, diff }) => {
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
            StateRef::State(state) => (
                0isize,
                state.map.width as isize,
                0isize,
                state.map.height as isize,
            ),
            StateRef::Snapshot(SnapshotDiffRef { snapshot, diff }) => {
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
            StateRef::State(state) => {
                for y in 0..state.map.height {
                    for x in 0..state.map.width {
                        if state.map.tiles[y][x] == Tile::Agent(agent) {
                            return Some((x as isize, y as isize));
                        }
                    }
                }

                None
            }
            StateRef::Snapshot(SnapshotDiffRef { snapshot, diff }) => {
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
            StateRef::State(state) => state.inventory.0.get(&agent).unwrap().wood as usize,
            StateRef::Snapshot(SnapshotDiffRef { snapshot, diff }) => {
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
            StateRef::State(state) => state
                .inventory
                .0
                .values()
                .map(|inv| inv.wood)
                .sum::<isize>() as usize,
            StateRef::Snapshot(SnapshotDiffRef { snapshot, diff }) => {
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
            StateRef::State(state) => state
                .inventory
                .0
                .get(&agent)
                .map(|inv| inv.water)
                .unwrap_or_default(),
            StateRef::Snapshot(SnapshotDiffRef { snapshot, diff }) => diff
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

impl StateMut for StateRefMut<'_, Lumberjacks> {
    fn increment_time(&mut self) {
        match self {
            StateRefMut::Snapshot(SnapshotDiffRefMut { diff, .. }) => {
                diff.ctr += 1;
            }
            _ => {}
        }
    }

    fn set_tile(&mut self, x: isize, y: isize, tile: Tile) {
        match self {
            StateRefMut::State(state) => {
                if x >= 0 && x < state.map.width as isize && y >= 0 && y < state.map.height as isize
                {
                    state.map.tiles[y as usize][x as usize] = tile;
                }
            }
            StateRefMut::Snapshot(SnapshotDiffRefMut { snapshot, diff }) => {
                if let Tile::Agent(agent) = tile {
                    diff.map.agents_pos.insert(agent, (x, y));
                }

                let (x, y) = (x - snapshot.map.left, y - snapshot.map.top);

                if x >= 0
                    && x < (config().agents.snapshot_radius * 2 + 1) as isize
                    && y >= 0
                    && y < (config().agents.snapshot_radius * 2 + 1) as isize
                {
                    if snapshot.map.tiles[y as usize][x as usize] == tile {
                        diff.map.tiles.remove(&(x, y));
                    } else {
                        diff.map.tiles.insert((x, y), tile);
                    }
                }
            }
        }
    }

    fn get_tile_ref_mut(&mut self, x: isize, y: isize) -> Option<&mut Tile> {
        match self {
            StateRefMut::State(state) => {
                if x >= 0 && x < state.map.width as isize && y >= 0 && y < state.map.height as isize
                {
                    Some(&mut state.map.tiles[y as usize][x as usize])
                } else {
                    None
                }
            }
            StateRefMut::Snapshot(SnapshotDiffRefMut { snapshot, diff }) => {
                let (x, y) = (x - snapshot.map.left, y - snapshot.map.top);

                if x >= 0
                    && x < (config().agents.snapshot_radius * 2 + 1) as isize
                    && y >= 0
                    && y < (config().agents.snapshot_radius * 2 + 1) as isize
                {
                    Some(
                        diff.map
                            .tiles
                            .entry((x, y))
                            .or_insert_with(|| snapshot.map.tiles[y as usize][x as usize]),
                    )
                } else {
                    None
                }
            }
        }
    }

    fn increment_inventory(&mut self, agent: AgentId) {
        match self {
            StateRefMut::State(state) => {
                state.inventory.0.entry(agent).or_default().wood += 1;
            }
            StateRefMut::Snapshot(SnapshotDiffRefMut { snapshot, diff }) => {
                // this is cumbersome because the diff has a real diff for the tree (+= diff.tree)
                // but for the water it is an override (= diff.water), so we need to fetch the
                // water from the snapshot when we create a new inventory diff
                diff.inventory.0.entry(agent).or_insert_with(
                    || AgentInventory {
                        wood: 0,
                        water: snapshot.inventory.0
                            .get(&agent)
                            .map(|inv| inv.water)
                            .unwrap_or_default()
                    }
                ).wood += 1;
            }
        }
    }

    fn decrement_inventory(&mut self, agent: AgentId) {
        match self {
            StateRefMut::State(state) => {
                state.inventory.0.get_mut(&agent).unwrap().wood -= 1;
            }
            StateRefMut::Snapshot(SnapshotDiffRefMut { snapshot, diff }) => {
                // this is cumbersome because the diff has a real diff for the tree (+= diff.tree)
                // but for the water it is an override (= diff.water), so we need to fetch the
                // water from the snapshot when we create a new inventory diff
                diff.inventory.0.entry(agent).or_insert_with(
                    || AgentInventory {
                        wood: 0,
                        water: snapshot.inventory.0
                            .get(&agent)
                            .map(|inv| inv.water)
                            .unwrap_or_default()
                    }
                ).wood -= 1;
            }
        }
    }

    fn set_water(&mut self, agent: AgentId, value: bool) {
        match self {
            StateRefMut::State(state) => {
                state.inventory.0.get_mut(&agent).unwrap().water = value;
            }
            StateRefMut::Snapshot(SnapshotDiffRefMut { snapshot, diff }) => {
                diff.inventory.0.entry(agent).or_default().water = value;
            }
        }
    }
}
