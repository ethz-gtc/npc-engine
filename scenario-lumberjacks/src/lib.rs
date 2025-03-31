/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::mem::MaybeUninit;
use std::path::PathBuf;
use std::sync::Once;
use std::{fs, io, mem, process};

use clap::{App, Arg};

use serde_json::Value;

mod behaviors;
mod config;
mod fitnesses;
mod game;
mod graph;
mod heatmap;
mod hooks;
mod inventory;
mod lumberjacks_domain;
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
pub use lumberjacks_domain::*;
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
                        if str.contains('=') {
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

                    let mut keys = k.split('.').peekable();

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
        #[allow(static_mut_refs)]
        mem::transmute(&NAME)
    }
}

pub fn config() -> &'static Config {
    unsafe {
        init();
        // Safe to transmute, initialized
        #[allow(static_mut_refs)]
        mem::transmute(&CONFIG)
    }
}

pub fn working_dir() -> &'static String {
    unsafe {
        init();
        // Safe to transmute, initialized
        #[allow(static_mut_refs)]
        mem::transmute(&WORKING_DIR)
    }
}

pub fn output_path() -> &'static String {
    unsafe {
        init();
        // Safe to transmute, initialized
        #[allow(static_mut_refs)]
        mem::transmute(&OUTPUT_PATH)
    }
}

pub fn batch() -> bool {
    unsafe {
        init();
        // Safe to transmute, initialized
        #[allow(static_mut_refs)]
        mem::transmute(BATCH)
    }
}
