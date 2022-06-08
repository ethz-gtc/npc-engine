/*
 *  SPDX-License-Identifier: Apache-2.0 OR MIT
 *  © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
 */

use std::fmt;
use std::{collections::HashMap, num::NonZeroU8};

use npc_engine_common::graphviz::GRAPH_OUTPUT_DEPTH;
use serde::de::{MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use npc_engine_common::{AgentId, StateDiffRef};

use crate::fitnesses;
use crate::Lumberjacks;

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct BatchConfig {
    #[serde(default)]
    pub runs: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        BatchConfig { runs: 1 }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GeneratorType {
    File { path: String },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct MapConfig {
    pub generator: GeneratorType,
    #[serde(default = "map_tree_height_default")]
    pub tree_height: NonZeroU8,
}

fn map_tree_height_default() -> NonZeroU8 {
    NonZeroU8::new(3).unwrap()
}

type Behaviors = HashMap<usize, (String, fn(StateDiffRef<Lumberjacks>, AgentId) -> f32)>;

#[derive(Serialize, Deserialize)]
#[serde(default, rename_all = "kebab-case")]
pub struct AgentsConfig {
    pub horizon_radius: usize,
    pub snapshot_radius: usize,
    pub tasks: bool,
    pub plan_others: bool,
    #[serde(
        deserialize_with = "behavior_deserializer",
        serialize_with = "behavior_serializer"
    )]
    pub(crate) behaviors: Behaviors,
}

impl fmt::Debug for AgentsConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentsConfig")
            .field("horizon_radius", &self.horizon_radius)
            .field("snapshot_radius", &self.snapshot_radius)
            .field("tasks", &self.tasks)
            .field("plan_others", &self.plan_others)
            .field(
                "behaviors",
                &self
                    .behaviors
                    .iter()
                    .map(|(k, (v, _))| (*k, v.clone()))
                    .collect::<HashMap<_, _>>(),
            )
            .finish()
    }
}

fn behavior_deserializer<'de, D>(deserializer: D) -> Result<Behaviors, D::Error>
where
    D: Deserializer<'de>,
{
    struct BehaviorVisitor;

    impl<'de> Visitor<'de> for BehaviorVisitor {
        type Value = Behaviors;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            write!(formatter, "a map of integers to behavior function names")
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            let mut out = HashMap::new();
            while let Some((k, v)) = map.next_entry::<usize, String>()? {
                let f = match v.as_str() {
                    "minimalist" => fitnesses::minimalist as _,
                    _ => panic!("unknown behavior specified"),
                };

                out.insert(k, (v, f));
            }
            Ok(out)
        }
    }

    deserializer.deserialize_map(BehaviorVisitor)
}

fn behavior_serializer<S>(behaviors: &Behaviors, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut map = serializer.serialize_map(Some(behaviors.len()))?;
    for (k, v) in behaviors {
        map.serialize_entry(k, &v.0)?;
    }
    map.end()
}

impl Default for AgentsConfig {
    fn default() -> Self {
        AgentsConfig {
            horizon_radius: 5,
            snapshot_radius: 10,
            tasks: false,
            plan_others: true,
            behaviors: Default::default(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "kebab-case")]
pub struct ActionWeightsConfig {
    pub barrier: f32,
    pub chop: f32,
    pub r#move: f32,
    pub plant: f32,
    pub refill: f32,
    pub wait: f32,
    pub water: f32,
}

impl Default for ActionWeightsConfig {
    fn default() -> Self {
        ActionWeightsConfig {
            barrier: 1.,
            chop: 20.,
            r#move: 10.,
            plant: 1.,
            refill: 20.,
            wait: 1.,
            water: 20.,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "kebab-case")]
pub struct MCTSConfig {
    pub visits: u32,
    pub exploration: f32,
    pub depth: u32,
    pub retention: f32,
    pub discount: f32,
    pub seed: Option<u64>,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        MCTSConfig {
            visits: 5000,
            exploration: 2f32.sqrt(),
            depth: 10,
            retention: 0.5,
            discount: 0.95,
            seed: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(default, rename_all = "kebab-case")]
pub struct FeaturesConfig {
    pub barriers: bool,
    pub teamwork: bool,
    pub watering: bool,
    pub planting: bool,
    pub waiting: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "kebab-case")]
pub struct AnalyticsConfig {
    pub metrics: bool,
    pub heatmaps: bool,
    pub graphs: bool,
    pub serialization: bool,
    pub screenshot: bool,
    pub performance: bool,
    pub graphs_depth: usize,
}
impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            metrics: Default::default(),
            heatmaps: Default::default(),
            graphs: Default::default(),
            serialization: Default::default(),
            screenshot: Default::default(),
            performance: Default::default(),
            graphs_depth: GRAPH_OUTPUT_DEPTH.load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "kebab-case")]
pub struct DisplayConfig {
    pub interactive: bool,
    pub background: (f32, f32, f32),
    pub padding: (usize, usize),
    pub inventory: bool,
}

impl Default for DisplayConfig {
    fn default() -> Self {
        DisplayConfig {
            interactive: true,
            background: (0., 0.44, 0.36),
            padding: (2, 2),
            inventory: true,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct Config {
    #[serde(default)]
    pub turns: Option<usize>,
    #[serde(default = "Default::default")]
    pub batch: BatchConfig,
    pub map: MapConfig,
    #[serde(default = "Default::default")]
    pub agents: AgentsConfig,
    #[serde(default = "Default::default")]
    pub action_weights: ActionWeightsConfig,
    #[serde(default = "Default::default")]
    pub mcts: MCTSConfig,
    #[serde(default = "Default::default")]
    pub features: FeaturesConfig,
    #[serde(default = "Default::default")]
    pub analytics: AnalyticsConfig,
    #[serde(default = "Default::default")]
    pub display: DisplayConfig,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct Experiment {
    pub base: String,
    #[serde(default = "Default::default")]
    pub trials: Value,
}
