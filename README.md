# NPC engine

A customizable Monte Carlo Tree Search (MCTS) planner with advanced featured tailored at multi-agent simulations and emergent narratives, by the [ETH Game Technology Center](https://gtc.inf.ethz.ch/research/emergent-narrative.html).

NPC engine provides the following features:

* domain-agnostic MCTS planner,
* varying-duration tasks, making it a scheduler in addition to a planner,
* heterogeneous agents (e.g. a world agent besides NPC agents), allowing for clean domain designs,
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

File [`tic-tac-toe.rs`](npc-engine-common/examples/tic-tac-toe.rs)

A traditional tic-tac-toe turn-based game to play interactively against the computer.

### Capture

File [`capture.rs`](npc-engine-common/examples/capture.rs)

A simulation of a competitive battle for capturing locations between two agents.

Agents can secure locations, collect ammunition and medical kits, and shoot each others.
This domain demonstrates actions of various durations, a world agent that respawns collectibles, disappearance of agents (upon death), and the use of the simple executor utility.

### Learn

File [`learn.rs`](npc-engine-common/examples/learn.rs)

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

### Ecosystem

Directory [`ecosystem`](npc-engine-common/examples/ecosystem/)

A 2-D ecosystem simulation in which herbivores and carnivores eat and die.

All agents plan in parallel in a multi-threaded way on their own partial world views.

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
