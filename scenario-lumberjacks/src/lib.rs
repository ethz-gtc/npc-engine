use std::collections::BTreeSet;
use std::mem::MaybeUninit;
use std::path::PathBuf;
use std::sync::Once;
use std::{fs, io, mem, process};

use clap::{App, Arg};

use npc_engine_turn::{AgentId, Behavior, Domain, SnapshotDiffRef};
use serde_json::Value;

mod behaviors;
mod config;
mod fitnesses;
mod game;
mod graph;
mod heatmap;
mod hooks;
mod inventory;
mod metrics;
mod screenshot;
mod serialization;
mod tasks;
mod tilemap;
mod util;
mod world;

pub use behaviors::*;
pub use config::*;
pub use game::*;
pub use graph::*;
pub use heatmap::*;
pub use hooks::*;
pub use inventory::*;
pub use metrics::*;
pub use screenshot::*;
pub use serialization::*;
pub use tasks::*;
pub use tilemap::*;
pub use util::*;
pub use world::*;

static INIT: Once = Once::new();
static mut CONFIG: MaybeUninit<Config> = MaybeUninit::uninit();
static mut WORKING_DIR: MaybeUninit<String> = MaybeUninit::uninit();
static mut OUTPUT_PATH: MaybeUninit<String> = MaybeUninit::uninit();
static mut NAME: MaybeUninit<String> = MaybeUninit::uninit();
static mut BATCH: MaybeUninit<bool> = MaybeUninit::uninit();

unsafe fn init() {
    INIT.call_once(|| {
        let matches = App::new("Lumberjacks")
            .version("1.0")
            .author("Sven Knobloch")
            .arg(
                Arg::with_name("config")
                    .required(true)
                    .help("Sets config file path"),
            )
            .arg(
                Arg::with_name("working-dir")
                    .required(false)
                    .takes_value(true)
                    .value_name("directory")
                    .long("working-dir")
                    .short("d")
                    .help("Overrides working dir"),
            )
            .arg(
                Arg::with_name("output")
                    .required(false)
                    .takes_value(true)
                    .value_name("directory")
                    .short("o")
                    .long("output")
                    .help("Sets output directory"),
            )
            .arg(
                Arg::with_name("name")
                    .required(false)
                    .takes_value(true)
                    .value_name("name")
                    .default_value("Lumberjacks")
                    .short("n")
                    .long("name")
                    .help("Sets name"),
            )
            .arg(
                Arg::with_name("batch")
                    .required(false)
                    .takes_value(false)
                    .short("b")
                    .long("batch")
                    .help("Enables batch mode"),
            )
            .arg(
                Arg::with_name("set")
                    .required(false)
                    .takes_value(true)
                    .multiple(true)
                    .number_of_values(1)
                    .short("s")
                    .long("set")
                    .validator(|str| {
                        if str.contains("=") {
                            Ok(())
                        } else {
                            Err("Invalid format, should be \"some.path=value\"".to_owned())
                        }
                    })
                    .help("Manually override a value in the config"),
            )
            .get_matches();

        let config_path = matches.value_of("config").unwrap();
        let config_dir = {
            let mut path = PathBuf::from(config_path);
            path.pop();
            path.to_str().unwrap().to_owned()
        };

        NAME = MaybeUninit::new(matches.value_of("name").unwrap().to_owned());

        OUTPUT_PATH =
            MaybeUninit::new(matches.value_of("output").unwrap_or(&config_dir).to_owned());

        WORKING_DIR = MaybeUninit::new(
            matches
                .value_of("working-dir")
                .unwrap_or(&config_dir)
                .to_owned(),
        );

        BATCH = MaybeUninit::new(matches.is_present("batch"));

        CONFIG = MaybeUninit::new({
            let mut json: Value = match config_path {
                "-" => {
                    let stdin = io::stdin();
                    serde_json::from_reader(stdin.lock()).unwrap()
                }
                path => {
                    let config_file = match fs::OpenOptions::new().read(true).open(path) {
                        Ok(file) => file,
                        Err(e) => {
                            println!("Cannot open config file {}: {}", path, e);
                            process::exit(1);
                        }
                    };
                    serde_json::from_reader(&config_file).unwrap()
                }
            };

            if let Some(values) = matches.values_of("set") {
                values.for_each(|value| {
                    let split = value.split('=').collect::<Vec<_>>();
                    let k = split[0];
                    let v = split[1];

                    let mut object = &mut json;

                    let mut keys = k.split(".").peekable();

                    while let Some(key) = keys.next() {
                        if keys.peek().is_some() {
                            // Path, get next map
                            let map = object
                                .as_object_mut()
                                .ok_or_else(|| format!("Invalid 'set' path: {}", k))
                                .unwrap();

                            // Key is not present or an object
                            if !map.contains_key(key) || map.get(key).unwrap().as_object().is_none()
                            {
                                map.insert(key.to_owned(), Value::Object(Default::default()));
                            }

                            object = map.get_mut(key).unwrap();
                        } else {
                            // Last element, insert into map
                            let map = object
                                .as_object_mut()
                                .ok_or_else(|| format!("Invalid 'set' path: {}", k))
                                .unwrap();

                            map.insert(
                                key.to_owned(),
                                serde_json::from_str(v)
                                    .map_err(|e| format!("'set' variable not valid: {}", e))
                                    .unwrap(),
                            );
                        }
                    }
                })
            }

            serde_json::from_value(json).unwrap()
        });
    })
}

pub fn name() -> &'static String {
    unsafe {
        init();
        // Safe to transmute, initialized
        mem::transmute(&NAME)
    }
}

pub fn config() -> &'static Config {
    unsafe {
        init();
        // Safe to transmute, initialized
        mem::transmute(&CONFIG)
    }
}

pub fn working_dir() -> &'static String {
    unsafe {
        init();
        // Safe to transmute, initialized
        mem::transmute(&WORKING_DIR)
    }
}

pub fn output_path() -> &'static String {
    unsafe {
        init();
        // Safe to transmute, initialized
        mem::transmute(&OUTPUT_PATH)
    }
}

pub fn batch() -> bool {
    unsafe {
        init();
        // Safe to transmute, initialized
        mem::transmute(BATCH)
    }
}

pub struct Lumberjacks;

impl Domain for Lumberjacks {
    type State = WorldState;
    type Snapshot = WorldSnapshot;
    type Diff = WorldDiff;
    type DisplayAction = Action;

    fn list_behaviors() -> &'static [&'static dyn Behavior<Self>] {
        &[&Human, &Lumberjack]
    }

    fn derive_snapshot(state: &Self::State, agent: AgentId) -> Self::Snapshot {
        let (x, y) = StateRef::State(state).find_agent(agent).unwrap();

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

        WorldSnapshot {
            inventory: InventorySnapshot(state.inventory.0.clone()),
            map,
        }
    }

    fn apply(state: &mut Self::State, snapshot: &Self::Snapshot, diff: &Self::Diff) {
        for (agent, AgentInventory { wood, water }) in &diff.inventory.0 {
            if let Some(inventory) = state.inventory.0.get_mut(agent) {
                inventory.wood = inventory.wood + *wood;
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
    fn get_current_value(snapshot: SnapshotDiffRef<Self>, agent: AgentId) -> f32 {
        // FIXME: cleanup compat code
        let state = StateRef::Snapshot(snapshot);
        if let Some((_, f)) = config().agents.behaviors.get(&(agent.0 as usize)) {
            f(state, agent)
        } else {
            state.get_inventory(agent) as f32
        }
    }

    fn update_visible_agents(snapshot: SnapshotDiffRef<Self>, agent: AgentId, agents: &mut BTreeSet<AgentId>) {
        // FIXME: cleanup compat code
        let state = StateRef::Snapshot(snapshot);
        if let Some((x, y)) = state.find_agent(agent) {
            if config().agents.plan_others {
                agents.extend(
                    state
                        .find_nearby_agents(x, y, config().agents.horizon_radius)
                        .into_iter(),
                );
            } else {
                agents.insert(agent);
            }
        } else {
            unreachable!("{:?}", snapshot);
        }
    }
}
