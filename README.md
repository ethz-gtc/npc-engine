# NPC engine

A customizable Monte Carlo Tree Search (MCTS) planner with advanced featured tailored to multi-agent simulations and emergent narratives, by the [ETH Game Technology Center](https://gtc.inf.ethz.ch/research/emergent-narrative.html).

NPC engine provides the following features:

* domain-agnostic MCTS planner,
* varying-duration tasks, making it a scheduler in addition to a planner,
* heterogeneous agents (e.g. a global bookkeeping agent along regular agents), allowing for clean domain designs,
* support for dynamically appearing and disappearing agents,
* choice of behaviour when tasks become invalid: abort sub-tree (as in board games), re-plan (as in simulations),
* custom state value function besides built-in rollout,
* multiple debugging features including tracing and search trees as graph outputs using graphviz's dot format,
* batteries included with several [examples](npc-engine-common/examples/), helper library and the [code of our research paper](scenario-lumberjacks/) (see below).

## Examples

We provide several examples that illustrate the various features of the engine.

To run an example, with [Rust installed](https://www.rust-lang.org/tools/install), type:

```
cargo run --release --example NAME
```

where NAME is one of the followings, in lowercase:

### Tic-tac-toe

```
cargo run --release --example tic-tac-toe
```

A traditional tic-tac-toe turn-based game to play interactively against the computer.

File [`tic-tac-toe.rs`](npc-engine-common/examples/tic-tac-toe.rs)

### Capture

```
cargo run --release --example capture
```

A simulation of a competitive battle for capturing locations between two agents.

Agents can secure locations, collect ammunition and medical kits, and shoot each others.
This domain demonstrates actions of various durations, a world agent that respawns collectibles, disappearance of agents (upon death), and the use of the simple executor utility.

File [`capture.rs`](npc-engine-common/examples/capture.rs)

### Learn

```
cargo run --release --example learn
```

A 1-D woodcutter simulation where the agent's performance improves over time due to self learning.

An agent must collect wood in a 1-D world, while using a low number of visits for the MCTS.
The state value function estimator is not the traditional roll-out simulation, but instead a feed-forward neural network with one hidden layer containing two neurons.
The simulation is repeated over multiple epochs, each time using the MCTS-based state value estimation to train the neural network for the next epoch using back-propagation.
This simulations shows that on the course of some hundreads of epochs, the performance of the agent — the amount of wood collected during a certain duration — improves by more than 50 %.

With Python 3, `scipy` and `matplotlib` installed, the performance over epochs, averaged over 20 runs, can be seen with the following command:

```
npc-engine-common/examples/plot-learn.py
```

The curve should look like that:

![Wood collected over epochs](images/learn_wood_collected_over_epochs.png)

File [`learn.rs`](npc-engine-common/examples/learn.rs)

### Ecosystem

```
cargo run --release --example ecosystem
```

A 2-D ecosystem simulation in which herbivores and carnivores eat and die.

All agents plan in parallel in a multi-threaded way on their own partial world views.

Directory [`ecosystem`](npc-engine-common/examples/ecosystem/)


### Lumberjack

Directory [`scenario-lumberjacks`](scenario-lumberjacks/)

The research code used in the paper [*Leveraging efficient planning and lightweight agent definition: a novel path towards emergent narrative* by Raymond et al, 2020](https://www.research-collection.ethz.ch/handle/20.500.11850/439084).
This domain features two lumberjacks whose value function is the total collected wood logs.
There is no explicit communication between these two agents, but because they plan for each others, a theory of mind emerges.

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

The NPC engine is composed of two packages: `npc-engine-common` and `npc-engine-utils`.
The documentation can be generated and browsed interactively with the following command:

```
cargo doc --open -p npc-engine-common -p npc-engine-utils
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
