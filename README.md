# NPC engine

A customizable Monte Carlo Tree Search (MCTS) planner with advanced featured tailored at multi-agent simulations and emergent narratives, by [ETH Game Technology Center](https://gtc.inf.ethz.ch/research/emergent-narrative.html).

NPC engine provides the following features:

* domain-agnostic MCTS planner,
* varying-duration tasks, making it a scheduler in addition to a planner,
* heterogeneous agents (e.g. a world agent besides NPC agents), allowing for clean domain designs,
* support for dynamically appearing and disappearing agents,
* choice of behaviour when tasks become invalid: abort sub-tree (as in board games), re-plan (as in simulations),
* custom state value function besides built-in rollout,
* multiple debugging features including tracing and search trees as graph outputs using graphviz's dot format,
* batteries included with several [examples](npc-engine-common/examples/), helper library and the [code of our research paper](scenario-lumberjacks/) (see below).

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