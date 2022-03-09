use std::{fmt, collections::{BTreeMap, HashMap, BTreeSet}};
use nonmax::NonMaxU8;
#[macro_use]
extern crate lazy_static;

use npc_engine_common::{AgentId, Task, StateDiffRef, impl_task_boxed_methods, StateDiffRefMut, Domain, IdleTask, TaskDuration, Behavior, AgentValue, ActiveTask, MCTSConfiguration, MCTS, graphviz, ActiveTasks};
use npc_engine_utils::{plot_tree_in_tmp, run_simple_executor, ExecutorCallbacks, ExecutableDomain, OptionDiffDomain};

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
struct Location(NonMaxU8);
impl Location {
	pub fn new(location: u8) -> Self {
		Self(NonMaxU8::new(location).unwrap())
	}
}

struct Map {
	links: HashMap<Location, HashMap<Location, u64>>,
	capture_locations: Vec<Location>
}
impl Map {
	fn ammo_location(&self) -> Location {
		Location::new(1)
	}
	fn medkit_location(&self) -> Location {
		Location::new(5)
	}
	fn path_len(&self, from: Location, to: Location) -> Option<u64> {
		self.links
			.get(&from)
			.and_then(|ends|
				ends.get(&to).copied()
			)
	}
	fn neighbors(&self, from: Location) -> Vec<Location> {
		self.links
			.get(&from)
			.map_or(Vec::new(), |ends|
				ends.keys().copied().collect()
			)
	}
	fn is_path(&self, from: Location, to: Location) -> bool {
		self.path_len(from, to).is_some()
	}
	fn capture_location(&self, index: u8) -> Location {
		self.capture_locations[index as usize]
	}
	fn capture_locations_count(&self) -> u8 {
		self.capture_locations.len() as u8
	}
}
lazy_static! {
    static ref MAP: Map = {
		let links_data = [
			(0, 1, 2),
			(0, 3, 2),
			(1, 2, 2),
			(1, 4, 1),
			(2, 5, 2),
			(3, 4, 1),
			(4, 5, 1),
			(3, 6, 2),
			(5, 6, 2)
		];
		let capture_locations = [
			0, 2, 6
		];
		let mut links = HashMap::<Location, HashMap<Location, u64>>::new();
		for (start, end, length) in links_data {
			let start = Location(NonMaxU8::new(start).unwrap());
			let end = Location(NonMaxU8::new(end).unwrap());
			// add bi-directional link
			links.entry(start)
				.or_default()
				.insert(end, length);
			links.entry(end)
				.or_default()
				.insert(start, length);
		}
		let capture_locations = capture_locations
			.map(|location|
				Location(NonMaxU8::new(location).unwrap())
			)
			.into();
		Map { links, capture_locations }
	};
}
const MAX_HP: u8 = 2;
const MAX_AMMO: u8 = 4;
const CAPTURE_DURATION: TaskDuration = 1;
const RESPAWN_AMMO_DURATION: u8 = 2;
const RESPAWN_MEDKIT_DURATION: u8 = 17;

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
struct AgentState {
	/// accumulated capture points for this agent
	acc_capture: u16,
	/// current or last location of the agent (if travelling),
	cur_or_last_location: Location,
	/// next location of the agent, if travelling, none otherwise
	next_location: Option<Location>,
	/// health point of the agent
	hp: u8,
	/// ammunition carried by the agent
	ammo: u8,
}

#[derive(Clone, Hash, PartialEq, Eq)]
enum CapturePointState {
	Free,
	Capturing(AgentId),
	Captured(AgentId)
}
impl fmt::Debug for CapturePointState {
	fn fmt(&self, f: &'_ mut fmt::Formatter) -> fmt::Result {
		match &self {
			CapturePointState::Free => f.write_str("__"),
			CapturePointState::Capturing(agent) => f.write_fmt(format_args!("C{:?}", agent.0)),
			CapturePointState::Captured(agent) => f.write_fmt(format_args!("H{:?}", agent.0)),
		}
	}
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
struct State {
	/// active agents
	agents: BTreeMap<AgentId, AgentState>,
	/// capture points
	capture_points: [CapturePointState; 3],
	/// ammo available at collection point
	ammo: u8,
	/// tick when ammo was collected
	ammo_tick: u8,
	/// medical kit available at collection point
	medkit: u8,
	/// tick when med kit was collected
	medkit_tick: u8
}
type Diff = Option<State>; // if Some, use this diff, otherwise use initial state

enum DisplayAction {
	Wait,
	Pick,
	Shoot(AgentId),
	StartCapturing(Location),
	Capturing(Location),
	StartMoving(Location),
	Moving(Location),
	WorldStep
}
impl Default for DisplayAction {
	fn default() -> Self {
		Self::Wait
	}
}
impl fmt::Debug for DisplayAction {
	fn fmt(&self, f: &'_ mut fmt::Formatter) -> fmt::Result {
		match &self {
			Self::Wait => f.write_str("Wait"),
			Self::Pick => f.write_str("Pick"),
			Self::Shoot(target) => f.write_fmt(format_args!("Shoot {:?}", target)),
			Self::StartCapturing(loc) => f.write_fmt(format_args!("StartCapturing {:?}", loc)),
			Self::Capturing (loc)=> f.write_fmt(format_args!("Capturing {:?}", loc)),
			Self::StartMoving(loc) => f.write_fmt(format_args!("StartMoving {:?}", loc)),
			Self::Moving(loc) => f.write_fmt(format_args!("Moving {:?}", loc)),
			Self::WorldStep => f.write_str("WorldStep")
		}
	}
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
struct Pick;
impl Task<CaptureGame> for Pick {
	fn duration(&self, _tick: u64, _state_diff: StateDiffRef<CaptureGame>, _agent: AgentId) -> TaskDuration {
		// Pick is instantaneous
		0
	}

	fn weight(&self, _tick: u64, _state_diff: StateDiffRef<CaptureGame>, _agent: AgentId) -> f32 {
        10.0
    }

	fn execute(
        &self,
        tick: u64,
        state_diff: StateDiffRefMut<CaptureGame>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<CaptureGame>>> {
		let diff = CaptureGame::get_cur_state_mut(state_diff);
		let agent_state = diff.agents.get_mut(&agent).unwrap();
		let location = agent_state.cur_or_last_location;
		match location {
			_ if location == MAP.ammo_location() => {
				agent_state.ammo = (agent_state.ammo + 1).min(MAX_AMMO);
				diff.ammo = 0;
				diff.ammo_tick = (tick & 0xff) as u8;
			},
			_ if location == MAP.medkit_location() =>  {
				agent_state.hp = (agent_state.hp + 1).min(MAX_HP);
				diff.medkit = 0;
				diff.medkit_tick = (tick & 0xff) as u8;
			},
			_ => unimplemented!()
		}
		// After Pick, the agent must wait one tick.
		Some(Box::new(IdleTask))
	}

	fn display_action(&self) -> <CaptureGame as Domain>::DisplayAction {
        DisplayAction::Pick
    }

	fn is_valid(&self, _tick: u64, state_diff: StateDiffRef<CaptureGame>, agent: AgentId) -> bool {
		let state = CaptureGame::get_cur_state(state_diff);
		state.agents
			.get(&agent)
			.map_or(false, |agent_state| {
				// We cannot pick while moving.
				if agent_state.next_location.is_some() {
					false
				} else {
					// We must be at a location where there is something to pick.
					let location = agent_state.cur_or_last_location;
					(location == MAP.ammo_location() && state.ammo > 0) ||
					(location == MAP.medkit_location() && state.medkit > 0)
				}
			})
    }

    impl_task_boxed_methods!(CaptureGame);
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
struct Shoot(AgentId);
impl Task<CaptureGame> for Shoot {
	fn duration(&self, _tick: u64, _state_diff: StateDiffRef<CaptureGame>, _agent: AgentId) -> TaskDuration {
		// Shoot is instantaneous
		0
	}

	fn weight(&self, _tick: u64, _state_diff: StateDiffRef<CaptureGame>, _agent: AgentId) -> f32 {
        10.0
    }

	fn execute(
        &self,
        _tick: u64,
        state_diff: StateDiffRefMut<CaptureGame>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<CaptureGame>>> {
		let diff = CaptureGame::get_cur_state_mut(state_diff);
		let agent_state = diff.agents.get_mut(&agent).unwrap();
		agent_state.ammo -= 1;
		let target_state = diff.agents.get_mut(&self.0).unwrap();
		if target_state.hp > 0 {
			target_state.hp -= 1;
		}
		if target_state.hp == 0 {
			diff.agents.remove(&self.0);
		}
		// After Shoot, the agent must wait one tick.
		Some(Box::new(IdleTask))
	}

	fn display_action(&self) -> <CaptureGame as Domain>::DisplayAction {
        DisplayAction::Shoot(self.0)
    }

	fn is_valid(&self, _tick: u64, state_diff: StateDiffRef<CaptureGame>, agent: AgentId) -> bool {
		let state = CaptureGame::get_cur_state(state_diff);
		state.agents
			.get(&agent)
			.map_or(false, |agent_state| {
				// We must have ammo to shoot.
				// We cannot shoot while moving.
				if agent_state.ammo == 0 || agent_state.next_location.is_some() {
					false
				} else {
					let location = agent_state.cur_or_last_location;
					// Target must exist.
					let target = state.agents.get(&self.0);
					target.map_or(false, |target| {
						// Target must be on our location or its adjacent paths.
						target.cur_or_last_location == location ||
						target.next_location.map_or(false,
							|next_location| next_location == location
						)
					})
				}
			})
    }

    impl_task_boxed_methods!(CaptureGame);
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
struct StartCapturing(u8);
impl Task<CaptureGame> for StartCapturing {
	fn duration(&self, _tick: u64, _state_diff: StateDiffRef<CaptureGame>, _agent: AgentId) -> TaskDuration {
		// StartCapture is instantaneous
		0
	}

	fn execute(
        &self,
        _tick: u64,
        state_diff: StateDiffRefMut<CaptureGame>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<CaptureGame>>> {
		let diff = CaptureGame::get_cur_state_mut(state_diff);
		diff.capture_points[self.0 as usize] = CapturePointState::Capturing(agent);
		Some(Box::new(Capturing(self.0)))
	}

	fn display_action(&self) -> <CaptureGame as Domain>::DisplayAction {
        DisplayAction::StartCapturing(MAP.capture_location(self.0))
    }

	fn is_valid(&self, _tick: u64, state_diff: StateDiffRef<CaptureGame>, agent: AgentId) -> bool {
		let state = CaptureGame::get_cur_state(state_diff);
		// if the point is already captured, we cannot restart capturing
		if state.capture_points[self.0 as usize] == CapturePointState::Captured(agent) {
			return false;
		}
		let capture_location = MAP.capture_location(self.0);
		state.agents
			.get(&agent)
			.map_or(false, |agent_state|
				// agent is at the right location and not moving
				agent_state.cur_or_last_location == capture_location &&
				agent_state.next_location.is_none()
			)
	}

	impl_task_boxed_methods!(CaptureGame);
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
struct Capturing(u8);
impl Task<CaptureGame> for Capturing {
	fn duration(&self, _tick: u64, _state_diff: StateDiffRef<CaptureGame>, _agent: AgentId) -> TaskDuration {
		// Capturing takes some time
		CAPTURE_DURATION
	}

	fn execute(
        &self,
        _tick: u64,
        state_diff: StateDiffRefMut<CaptureGame>,
        agent: AgentId,
    ) -> Option<Box<dyn Task<CaptureGame>>> {
		let diff = CaptureGame::get_cur_state_mut(state_diff);
		diff.capture_points[self.0 as usize] = CapturePointState::Captured(agent);
		None
	}

	fn display_action(&self) -> <CaptureGame as Domain>::DisplayAction {
        DisplayAction::Capturing(MAP.capture_location(self.0))
    }

	fn is_valid(&self, _tick: u64, state_diff: StateDiffRef<CaptureGame>, agent: AgentId) -> bool {
		let state = CaptureGame::get_cur_state(state_diff);
		state.agents.get(&agent).is_some() &&
		state.capture_points[self.0 as usize] == CapturePointState::Capturing(agent)
		// note: no need to check agent location, as this task is always a follow-up of StartCapturing
	}

	impl_task_boxed_methods!(CaptureGame);
}


#[derive(Clone, Hash, PartialEq, Eq, Debug)]
struct StartMoving {
	to: Location
}
impl Task<CaptureGame> for StartMoving {
	fn duration(&self, _tick: u64, _state_diff: StateDiffRef<CaptureGame>, _agent: AgentId) -> TaskDuration {
		// Start moving is instantaneous
		0
	}

	fn execute(
		&self,
		_tick: u64,
		state_diff: StateDiffRefMut<CaptureGame>,
		agent: AgentId,
	) -> Option<Box<dyn Task<CaptureGame>>> {
		let diff = CaptureGame::get_cur_state_mut(state_diff);
		let agent_state = diff.agents.get_mut(&agent).unwrap();
		let to = self.to;
		agent_state.next_location = Some(to);
		// After starting, the agent must complete the move.
		let from = agent_state.cur_or_last_location;
		Some(Box::new(Moving { from, to }))
	}

	fn display_action(&self) -> <CaptureGame as Domain>::DisplayAction {
		DisplayAction::StartMoving(self.to)
	}

	fn is_valid(&self, _tick: u64, state_diff: StateDiffRef<CaptureGame>, agent: AgentId) -> bool {
		let state = CaptureGame::get_cur_state(state_diff);
		state.agents
			.get(&agent)
			.map_or(false, |agent_state| {
				let location = agent_state.cur_or_last_location;
				// We must be at a location (i.e. not moving).
				agent_state.next_location.is_none() &&
				// There must be a path to target.
				MAP.is_path(location, self.to)
			})
	}

	impl_task_boxed_methods!(CaptureGame);
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
struct Moving {
	from: Location,
	to: Location
}
impl Task<CaptureGame> for Moving {
	fn duration(&self, _tick: u64, _state_diff: StateDiffRef<CaptureGame>, _agent: AgentId) -> TaskDuration {
		MAP.path_len(self.from, self.to).unwrap()
	}

	fn execute(
		&self,
		_tick: u64,
		state_diff: StateDiffRefMut<CaptureGame>,
		agent: AgentId,
	) -> Option<Box<dyn Task<CaptureGame>>> {
		let diff = CaptureGame::get_cur_state_mut(state_diff);
		let agent_state = diff.agents.get_mut(&agent).unwrap();
		agent_state.cur_or_last_location = self.to;
		agent_state.next_location = None;
		// The agent has completed the move, it is now idle.
		None
	}

	fn display_action(&self) -> <CaptureGame as Domain>::DisplayAction {
		DisplayAction::Moving(self.to)
	}

	fn is_valid(&self, _tick: u64, state_diff: StateDiffRef<CaptureGame>, agent: AgentId) -> bool {
		let state = CaptureGame::get_cur_state(state_diff);
		// This is a follow-up of StartMoving, so as the map is static, we assume
		// that as long as the agent exists, the task is valid.
		state.agents
			.get(&agent)
			.is_some()
	}

	impl_task_boxed_methods!(CaptureGame);
}


struct AgentBehavior;
impl Behavior<CaptureGame> for AgentBehavior {
	fn add_own_tasks(
		&self,
		tick: u64,
		state_diff: StateDiffRef<CaptureGame>,
		agent: AgentId,
		tasks: &mut Vec<Box<dyn Task<CaptureGame>>>,
	) {
		let state = CaptureGame::get_cur_state(state_diff);
		let agent_state = state.agents.get(&agent);
		if let Some(agent_state) = agent_state {
			// already moving, cannot do anything else
			if agent_state.next_location.is_some() {
				return;
			}
			tasks.push(Box::new(IdleTask));
			for to in MAP.neighbors(agent_state.cur_or_last_location) {
				let task = StartMoving{ to };
				tasks.push(Box::new(task));
			}
			let other_agent = if agent.0 == 0 { AgentId(1) } else { AgentId(0) };
			let other_tasks: Vec<Box<dyn Task<CaptureGame>>> = vec![
				Box::new(Pick),
				Box::new(Shoot(other_agent))
			];
			for task in other_tasks {
				if task.is_valid(tick, state_diff, agent) {
					tasks.push(task);
				}
			}
			for capture_index in 0..MAP.capture_locations_count() {
				let task = StartCapturing(capture_index);
				if task.is_valid(tick, state_diff, agent) {
					tasks.push(Box::new(task));
				}
			}
		}
	}

	fn is_valid(&self, _tick: u64, _state_diff: StateDiffRef<CaptureGame>, agent: AgentId) -> bool {
		agent != WORLD_AGENT_ID
	}
}


const WORLD_AGENT_ID: AgentId = AgentId(9); // 9 could be u32::MAX in a realistic scenario

fn respawn_timeout_ammo(now: u8, before: u8) -> bool {
	now.wrapping_sub(before) > RESPAWN_AMMO_DURATION
}
fn respawn_timeout_medkit(now: u8, before: u8) -> bool {
	now.wrapping_sub(before) > RESPAWN_MEDKIT_DURATION
}

#[derive(Clone, Hash, PartialEq, Eq, Debug)]
struct WorldStep;
impl Task<CaptureGame> for WorldStep {
	fn duration(&self, _tick: u64, _state_diff: StateDiffRef<CaptureGame>, _agent: AgentId) -> TaskDuration {
		1
	}

	fn execute(
		&self,
		tick: u64,
		state_diff: StateDiffRefMut<CaptureGame>,
		_agent: AgentId,
	) -> Option<Box<dyn Task<CaptureGame>>> {
		let diff = CaptureGame::get_cur_state_mut(state_diff);
		// for each captured point, increment the score of the corresponding agent
		for capture_point in &diff.capture_points {
			if let CapturePointState::Captured(agent) = capture_point {
				if let Some(agent_state) = diff.agents.get_mut(agent) {
					agent_state.acc_capture += 1;
				}
			}
		}
		// respawn if timeout
		let now = (tick & 0xff) as u8;
		if respawn_timeout_ammo(now, diff.ammo_tick) {
			diff.ammo = 1;
			diff.ammo_tick = now;
		}
		if respawn_timeout_medkit(now, diff.medkit_tick) {
			diff.medkit = 1;
			diff.medkit_tick = now;
		}

		Some(Box::new(WorldStep))
	}

	fn display_action(&self) -> <CaptureGame as Domain>::DisplayAction {
		DisplayAction::WorldStep
	}

	fn is_valid(&self, _tick: u64, _state_diff: StateDiffRef<CaptureGame>, _agent: AgentId) -> bool {
		true
	}

	impl_task_boxed_methods!(CaptureGame);
}

struct WorldBehavior;
impl Behavior<CaptureGame> for WorldBehavior {
	fn add_own_tasks(
		&self,
		_tick: u64,
		_state_diff: StateDiffRef<CaptureGame>,
		_agent: AgentId,
		tasks: &mut Vec<Box<dyn Task<CaptureGame>>>,
	) {
		tasks.push(Box::new(WorldStep));
	}

	fn is_valid(&self, _tick: u64, _state_diff: StateDiffRef<CaptureGame>, agent: AgentId) -> bool {
		agent == WORLD_AGENT_ID
	}
}

struct CaptureGame;
impl Domain for CaptureGame {
	type State = State;
	type Diff = Diff;
	type DisplayAction = DisplayAction;

	fn list_behaviors() -> &'static [&'static dyn Behavior<Self>] {
		&[&AgentBehavior, &WorldBehavior]
	}

	fn get_current_value(_tick: u64, state_diff: StateDiffRef<Self>, agent: AgentId) -> AgentValue {
		let value_zero: AgentValue = AgentValue::new(0.).unwrap();
		let state = Self::get_cur_state(state_diff);
		state.agents.get(&agent).map_or(
			value_zero,
			|agent_state| AgentValue::from(agent_state.acc_capture)
		)
	}

	fn update_visible_agents(_start_tick: u64, _tick: u64, state_diff: StateDiffRef<Self>, _agent: AgentId, agents: &mut BTreeSet<AgentId>) {
		let state = Self::get_cur_state(state_diff);
		agents.clear();		
		agents.extend(state.agents.keys());
		agents.insert(WORLD_AGENT_ID);
	}
	fn get_state_description(state_diff: StateDiffRef<Self>) -> String {
		let state = Self::get_cur_state(state_diff);
		let mut s = format!("World: ❤️ {} ({}), • {} ({}), ⚡: ", state.medkit, state.medkit_tick, state.ammo, state.ammo_tick);
		s += &(0..MAP.capture_locations_count())
			.map(|index|
				format!("{:?}", state.capture_points[index as usize])
			)
			.collect::<Vec<_>>()
			.join(" ");
		for (id, state) in &state.agents {
			if let Some(target) = state.next_location {
				s += &format!("\nA{} in {}-{}, ❤️ {}, • {}, ⚡{}", id.0, state.cur_or_last_location.0, target.0, state.hp, state.ammo, state.acc_capture);
			} else {
				s += &format!("\nA{} @    {}, ❤️ {}, • {}, ⚡{}", id.0, state.cur_or_last_location.0, state.hp, state.ammo, state.acc_capture);
			}
		}
		s
	}
}
impl OptionDiffDomain for CaptureGame {
	type Domain = CaptureGame;
	type State = <CaptureGame as Domain>::State;
}
impl ExecutableDomain for CaptureGame {
	fn apply_diff(diff: Self::Diff, state: &mut Self::State) {
		if let Some(diff) = diff {
			*state = diff;
		}
	}
}

struct CaptureGameCallbacks;
impl ExecutorCallbacks<CaptureGame> for CaptureGameCallbacks {

	fn create_initial_state() -> State {
		let agent0_id = AgentId(0);
		let agent0_state = AgentState {
			acc_capture: 0,
			cur_or_last_location: Location::new(0),
			next_location: None,
			hp: MAX_HP,
			ammo: 0 //MAX_AMMO,
		};
		let agent1_id = AgentId(1);
		let agent1_state = AgentState {
			acc_capture: 0,
			cur_or_last_location: Location::new(6),
			next_location: None,
			hp: MAX_HP,
			ammo: 0 //MAX_AMMO,
		};
		State {
			agents: BTreeMap::from([
				(agent0_id, agent0_state),
				(agent1_id, agent1_state),
			]),
			capture_points: [
				CapturePointState::Free,
				CapturePointState::Free,
				CapturePointState::Free
			],
			ammo: 1,
			ammo_tick: 0,
			medkit: 1,
			medkit_tick: 0
		}
	}

	fn init_task_queue() -> ActiveTasks<CaptureGame> {
		vec![
			ActiveTask::new_with_end(0, AgentId(0), Box::new(IdleTask)),
			ActiveTask::new_with_end(0, AgentId(1), Box::new(IdleTask)),
			ActiveTask::new_with_end(0, WORLD_AGENT_ID, Box::new(WorldStep)),
		].into_iter().collect()
	}

	fn keep_agent(state: &State, agent: AgentId) -> bool {
		agent == WORLD_AGENT_ID || state.agents.contains_key(&agent)
	}

	fn post_mcts_run_hook(mcts: &MCTS<CaptureGame>, last_active_task: &ActiveTask<CaptureGame>) {
		let time_text = format!("T{}", mcts.start_tick);
		let agent_id_text = format!("A{}", mcts.agent().0);
		let task_name = format!("{:?}", last_active_task.task);
		let last_task_name = task_name
			.replace(" ", "")
			.replace("(", "")
			.replace(")", "")
			.replace("{", "_")
			.replace("}", "")
			.replace(" ", "_")
			.replace(":", "_")
			.replace(",", "_")
		;
		if let Err(e) = plot_tree_in_tmp(
			mcts,
			"capture_graphs",
			&format!("{agent_id_text}-{time_text}-{last_task_name}")
		) {
			println!("Cannot write search tree: {e}");
		}
	}
}

fn main() {
	const CONFIG: MCTSConfiguration = MCTSConfiguration {
		allow_invalid_tasks: true,
		visits: 5000,
		depth: 50,
		exploration: 1.414,
		discount_hl: 17.,
		seed: None
	};
	graphviz::GRAPH_OUTPUT_DEPTH.store(7, std::sync::atomic::Ordering::Relaxed);
	use std::io::Write;
	env_logger::builder()
		.format(|buf, record|
			writeln!(buf, "{}", record.args())
		)
		.filter(None, log::LevelFilter::Info)
		.init();
	run_simple_executor::<CaptureGame, CaptureGameCallbacks>(&CONFIG);
}