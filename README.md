A work in progress. Currently, single-agent distributed PPO is used for traffic light control. Current envs are single intersection,
and multi-intersection. Very simple so far. Can run normally, grid-search or random-search from `main.py`. Just change the constants.
Can run a simple multi-agent scenario (or single agent) by changing "single_agent" to false in the constants. Right now, 
independent distributed PPO is used for the multi-agent setting, all parameters are shared (for both actor and critic) and reward is global.
Im working on adding features from this [paper](https://arxiv.org/abs/1903.04527) 
such as a discounted local/global reward and state based on intersection neighborhood.
