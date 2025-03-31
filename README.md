# NPC engine

**Core:** 
[![Crates.io][core-crates-badge]][core-crates-url]
[![Docs.rs][core-docs-badge]][core-docs-url]
[![Build Status][ci-badge]][ci-url]

**Utils:** 
[![Crates.io][utils-crates-badge]][utils-crates-url]
[![Docs.rs][utils-docs-badge]][utils-docs-url]
[![Build Status][ci-badge]][ci-url]

[core-crates-badge]: https://img.shields.io/crates/v/npc-engine-core
[core-crates-url]: https://crates.io/crates/npc-engine-core
[core-docs-badge]: https://img.shields.io/docsrs/npc-engine-core
[core-docs-url]: https://docs.rs/npc-engine-core
[utils-crates-badge]: https://img.shields.io/crates/v/npc-engine-utils
[utils-crates-url]: https://crates.io/crates/npc-engine-utils
[utils-docs-badge]: https://img.shields.io/docsrs/npc-engine-utils
[utils-docs-url]: https://docs.rs/npc-engine-utils

[ci-badge]: https://img.shields.io/github/actions/workflow/status/ethz-gtc/npc-engine/ci.yml?branch=main
[ci-url]: https://github.com/ethz-gtc/npc-engine/actions

© 2020-2022 ETH Zurich and other contributors. See [AUTHORS.txt](AUTHORS.txt) for more details.

A customizable [Monte Carlo Tree Search (MCTS)](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) planner with advanced featured tailored to multi-agent simulations and emergent narratives, by the [ETH Game Technology Center](https://gtc.inf.ethz.ch/research/emergent-narrative.html).

NPC engine provides the following features:

* domain-agnostic MCTS planner,
* varying-duration tasks, making it a scheduler in addition to a planner,
* heterogeneous agents (e.g. a global bookkeeping agent along regular agents), allowing for clean domain designs,
* support for dynamically appearing and disappearing agents (even within a single planning tree),
* choice of behavior when tasks become invalid: prune that subtree (as in board games), or re-plan (as in simulations),
* custom state value function in addition to standard rollout,
* multiple debugging features including tracing and plotting search trees as graphs using graphviz's dot format,
* batteries included with several [examples](npc-engine-core/examples/), helper library and the [code of our research paper](scenario-lumberjacks/) (see below).

The NPC engine is composed of two packages: [`npc-engine-core`](https://crates.io/crates/npc-engine-core) and [`npc-engine-utils`](https://crates.io/crates/npc-engine-utils).

## Getting it

To get the NPC engine, clone this repository:

```
git clone https://github.com/ethz-gtc/npc-engine.git
```

Make sure that you have GIT LFS installed, you can read [here how to install it](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).
If not, the PNG files will not be properly fetched, and you will see PNG decoding errors.

## Examples

We provide several examples that illustrate the various features of the engine.

To run the examples, you need to have [Rust installed](https://www.rust-lang.org/tools/install).

Make sure your console uses a font that supports unicode emojis, as we use emojis for visualization.
On recent Linux and MacOS machines, they are supported out of the box.
On Windows 10 and newer, you can install the recently-released [Windows Terminal](https://aka.ms/terminal), and use it with `wt`.


### Tic-tac-toe

```
cargo run --release --example tic-tac-toe
```

A traditional tic-tac-toe turn-based game to play interactively against the computer.
To make a move, type `X Y` where `X` and `Y` are the coordinates (0, 1, or 2) of your move.

Source directory: [`tic-tac-toe`](npc-engine-core/examples/tic-tac-toe/)

### Capture

```
cargo run --release --example capture
```

A simulation of a competitive battle between two agents in which each tries to capture locations.

Agents can secure locations, collect ammunition and medical kits, and shoot each others.
They plan in turn.
The simulation outputs the state of the world and the agents each time a task finishes:

* For the world, it shows the availability of med kit (❤️), ammunition (•) and the state of capture locations (⚡).
  For a capture location, "__" means that it belongs to no one, "CX" (X = 0 or 1) means that an agent is capturing it, and "HX" (X = 0 or 1) means that an agent is holding that location.
  A held location brings victory points over time.
* For the agents, donated "AX" with X being the agent id, the followings are shown: their location (e.g. 0) or locations (e.g. 0-1, in case of movement), their health points (❤️), the amount of ammunition they carry (•), and their number of victory points (⚡).
* The task about to be completed is shown, and if its execution fails because its completion pre-conditions have changed since planning, that is also shown.
* Then planning is run for that agent, and the chosen task is shown and queued for execution.

This domain demonstrates actions of various durations, a world agent that respawns collectibles, disappearance of agents (upon death), and the use of the simple executor utility.


Source directory: [`capture`](npc-engine-core/examples/capture/)

### Learn

```
cargo run --release --example learn
```

A 1-D woodcutter simulation where the agent's performance improves over time due to self learning. The amount of wood collected is output after each run.

An agent must collect wood by cutting down trees in a 1-D world, while using a low number of visits for the MCTS. 
The state value function estimator in this example is not a standard roll-out simulation, but instead a feed-forward neural network with one hidden layer containing two neurons.
The simulation is repeated over multiple epochs, each time using the MCTS-based state value estimation to train the neural network for the next epoch using back-propagation.
This simulation shows that over the course of several hundreds epochs, the performance of the agent — the amount of wood collected during a certain duration — improves by more than 50 %.

With Python 3, `scipy` and `matplotlib` installed, the performance over epochs, averaged over 20 runs, can be seen with the following command:

```
npc-engine-core/examples/learn/plot.py
```

The curve should look like this:

![Wood collected over epochs](images/learn_wood_collected_over_epochs.png)

Source directory: [`learn`](npc-engine-core/examples/learn/)

### Ecosystem

```
cargo run --release --example ecosystem
```

A 2-D ecosystem simulation in which herbivores (🐄) and carnivores (🐅) eat and die.

The world consists of a tilemap where each tile can be empty (dark green), an obstacle (gray), or grass (green).
A grass tile can provide 1-3 units of food, visualized with increasing saturation levels.
By eating, a herbivore reduces the amount of food of the tile it's standing on by 1.
Herbivores are born with a given units of food, and can store a limited units of food.
Carnivores can eat herbivores on an adjacent tile or one tile away.
They can also jump to one tile away, including over other agents, but not over obstacles, at a cost of 1 additional unit of food.
Carnivores are born with a given units of food, and can store a limited units of food.
When agents eat, their stored food are resplenished to their maximum amount.
Periodically, all agents consume one unit of food.
If they do not have food any more, they die.

All agents plan in parallel in a multi-threaded way on their own partial world views.
The planning lasts a fixed number of frames, and other actions are instantaneous.
Agents see the map up to a limited distance and also consider other agents up to a limited distance, typically lower than the first one.
When planning, agents only consider a limited number of closest other agents, event if it sees more than these.
Agents aim at a given number of visits per planning, but if not enough computational power is available, planning might end earlier.
In that case, the plan quality degrades.
The simulation targets a certain number of frames per second.
A minimum number of visits per agents is ensured, if there is not enough computational power even for these, the simulation will slow down.

A world agent, with only one "step" action, periodically decreases the amount of food per animal agent, regrows the grass, and spawn animals' children.
These periods differ.

The exact domain, planning and simulation parameters are available in file [`constants.rs`](npc-engine-core/examples/ecosystem/constants.rs).

The simulation stops if all agents die.
After running the simulation, one can plot the statistics of the ecosystem over time with the following command (assuming you have Python3, Numpy and Matplotlib installed):

```
npc-engine-core/examples/ecosystem/plot_ecosystem_stats.py
```

Source directory: [`ecosystem`](npc-engine-core/examples/ecosystem/)


### Lumberjack

Directory [`scenario-lumberjacks`](scenario-lumberjacks/)

The research code used in the paper [*Leveraging efficient planning and lightweight agent definition: a novel path towards emergent narrative* by Raymond et al, 2020](https://www.research-collection.ethz.ch/handle/20.500.11850/439084).
This domain features two lumberjacks whose value function is the total collected wood logs.
There is no explicit communication between these two agents, but because they plan for each others, a theory of mind emerges.
You can get an overview of this work in [this video](https://youtu.be/-O_iOwNVGDw).

To see a basic run of this scenario, use the following command:

```
cargo run --release --bin lumberjacks scenario-lumberjacks/experiments/base.json
```

You can step the simulation by pressing "space".
There are two agents, red and yellow, which execute their actions turn by turn.
In this basic scenario, the red agent will collect all wood at the right because it reasons about the possible actions of the yellow agent and thus collects the trees in the optimal order.

The scenario can be run non-interactively by setting a configuration parameter with the `-s` flag:
```
cargo run --release --bin lumberjacks -- -s display.interactive=false scenario-lumberjacks/experiments/base.json
```

Here are some additional interesting experiments from the paper:

#### Basic competition

The red agent collects the wood in an order that prevents the yellow agent from collecting any, leaving all for itself.

```
cargo run --release --bin lumberjacks scenario-lumberjacks/experiments/competition-basic/base.json
```

#### Advanced competition

The red agent typically invests the first wood it collects to build a barrier to block the yellow agent from collecting any more wood, leaving more for itself.
This shows the ability to reason about others and accept a short-term loss for a bigger, longer-term profit.

```
cargo run --release --bin lumberjacks scenario-lumberjacks/experiments/barrier/base.json
```

#### Cooperation

Now all adjacent agents to a tree being cut also receive one wood.
Because of that, and implicitly through planning, the two agents synchronize together to cut the same tree.

```
cargo run --release --bin lumberjacks scenario-lumberjacks/experiments/teamwork-basic/base.json
```

#### Sustainability

A well is available for the agent to water a tree and let it regrow to full size.
The agent does not cut the tree fully, but goes to fetch the water from the well when it is almost dead.

```
cargo run --release --bin lumberjacks scenario-lumberjacks/experiments/optimization/base.json
```

If we also allow the agent to plant a new tree (using one wood), the agent cuts the tree and replants one closer to the well:

```
cargo run --release --bin lumberjacks -- -s features.planting=true scenario-lumberjacks/experiments/optimization/base.json
```

## Documentation

The NPC engine is composed of two packages: `npc-engine-core` and `npc-engine-utils`.
The documentation can be generated and browsed interactively with the following command:

```
cargo doc --open -p npc-engine-core -p npc-engine-utils
```

### A note on performance

Rust is heavily dependent on compiler optimizations.
Make sure that you include the `--release` flag to your cargo call for efficient execution.

## Generate PDF of search tree graphs

Some examples (ecosystem, capture, tic-tac-toe) use the the helper functions `plot_tree_in_tmp_with_task_name` and `plot_tree_in_tmp` to generate graphs of the search tree in the [Graphviz's dot format](https://graphviz.org/) in your temporary directory. Using the generated `.dot` files, you can create PDF trees with the following command:

```
for file in $(ls *.dot); do dot -Tpdf $file -o `basename -s .dot $file`.pdf; done
```

## Please cite us

If you use this software in an academic context, please cite our paper:

```
@inproceedings{raymond2020leveraging,
	title={Leveraging efficient planning and lightweight agent definition: a novel path towards emergent narrative},
	author={Raymond, Henry and Knobloch, Sven and Z{\"u}nd, Fabio and Sumner, Robert W and Magnenat, St{\'e}phane},
	booktitle={12th Intelligent Narrative Technolgies Workshop, held with the AIIDE Conference (INT10 2020)},
	doi={10.3929/ethz-b-000439084},
	year={2020},
}
```

## Thanks and Alternatives

We would like to thank Patrick Eppensteiner, Nora Tommila, and Heinrich Grattenthaler for their contributions to this research project.

Some possible alternatives to this work, also in Rust, are the [mcts](https://crates.io/crates/mcts), [arbor](https://crates.io/crates/arbor) and [board-game](https://crates.io/crates/board-game) frameworks.

## License

NPC Engine is free and open source! All code in this repository is dual-licensed under either:

* MIT License ([LICENSE-MIT](docs/LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))
* Apache License, Version 2.0 ([LICENSE-APACHE](docs/LICENSE-APACHE) or [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.