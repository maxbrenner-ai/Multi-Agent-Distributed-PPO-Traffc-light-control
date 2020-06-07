# Distributed PPO for Traffic Light Control
Uses a distributed version of the deep reinforcement learning algorithm [PPO](https://arxiv.org/abs/1707.06347) to control a grid of traffic lights for optimized traffic flow through the system. The traffic enviornment is implemented in the realistic traffic simulation [SUMO](https://sumo.dlr.de/docs/index.html). Multi-agent RL (MARL) is implemented with each traffic light acting as a single agent. 

## SUMO / Traci
SUMO (**S**imulation of **U**rban **MO**bility) is a continuous road traffic simulation. [TraCI](Thttps://sumo.dlr.de/docs/TraCI.html) (**Tra**ffic **C**ontrol **I**nterface) connects to a SUMO simulation in a programming language (in this case Python) to allow for feeding inputs and recieving outputs. 

![SUMO picture](/images/sumo.png)

The environments implemented for this problem are grids where an intersection is controlled by a traffic light. Either NS cars can go or EW cars, at a time. So each intersection has 2 possible configurations. Cars spawn at the edges and then have a predefined destination edge where they despawn.

## Models
### PPO
[Proximal Policy Optimization](https://openai.com/blog/openai-baselines-ppo/) (PPO) is a policy gradient based reinforcement learning algorithm created by OpenAI. It is efficient and fairly simple and tends to be the goto for RL nowadays. There are a lot of great tutorials and code on PPO ([this](https://medium.com/@jonathan_hui/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12), [this](https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/agent/PPO_agent.py) and many more). 

![PPO code](/images/ppo.png)

### DPPO
Distributed algorithms use multiple processes to speed up existing algorithms such as PPO. There arent as many simple resources on DPPO but I used a few different sources noted in my code such as [this repo](https://github.com/alexis-jacq/Pytorch-DPPO). I first implemented single-agent RL which means that in a single environment there is only one agent. In this apps case, this means all traffic lights are controlled by one agent. However, that means as the grid size increases the action size increases exponentially. 

###  MARL
![1x1 grid](/images/1_1-grid.png)

For example, the action space for a single intersection is 2 as either the NS light can be green or the EW light can be greed. 

![2x2 grid](/images/2_2-grid.png)

The number of actions for a 2x2 grid is 2^4 = 16. For example if 1 means NS is green and 0 means EW is green. Then 1011 in binary (13 in decimal) would mean that 3 of the 4 intersections are NS green. This can become a problem as the grid gets even larger. 

![MARL](/images/marl.png)

[Cooperative MARL](https://arxiv.org/abs/1908.03963) is a way to fix this ["curse of dimensionality"](https://en.wikipedia.org/wiki/Curse_of_dimensionality) problem. With MARL there are multiple agents in the environment. And in this case each agent controls a single intersection. So now an agent only has 2 possible actions no matter how big the grid gets! MARL also helps with inputs. Instead of a single agent needing to be trained to deal with say 4 states (for a 2x2 grid) it can just deal with one. MARL is a great tool in cases where your problem can run into scaling issues. 

In the case of this repo, I use independent MARL which means each agent does not directly communicate. However, each actor and critic share parameters across all agents. One trick for better cooperation is to share certain info across agents (other than just weights). Reward and states are two popular items to share. This [post](https://bair.berkeley.edu/blog/2018/12/12/rllib/) by Berkeley goes into this more.

## How to Run this
### Depndencies
* numpy
* traci
* sumolib
* scipy
* pytorch
* pandas

### Running
Can alter `constants.json` or `constans-grid.json` in /constants to change different hyperparameters. In `main.py` can run experiments with `run_normal` (runs multiple experiments using `constants.json`), `run_random_search` (runs a random search on `constants-grid.json`) or `run_grid_search` (runs a grid search on `constants-grid.json`). Can save and load models. Can also visualize models by running `vis_agent.py` and changing `run(load_model_file=<MODEL FILE NAME>)` to the model file. The 4 envs implemented are 1x1, 2x2, 3x3 and 4x4. 

`shape` is the grid, `rush_hour` can be set to true for 2x2 which adds a kind of rush-hour spawning probability distribution. And `uniform_generation_probability` is the spawn rate for cars when `rush_hour` is false. 
```
"environment": {
        "shape": [4, 4],
        "rush_hour": false,
        "uniform_generation_probability": 0.06
    },
```

Change `num_workers` based on how many processes you want active for the distribibuted part of DPPO. 
```
    "parallel":{
        "num_workers": 8
    }
```
Finally, you can change the `agent_type` to `rule` if you want a simple rule based agent to run (which just changes each light after a set amount of time). And can change `single_agent` to true to not use MARL. 

```
    "agent": {
        "agent_type": "ppo",
        "single_agent": false
    },
```
