# Distributed PPO for Traffic Light Control
Uses a distributed version of the deep reinforcement learning algorithm [PPO](https://arxiv.org/abs/1707.06347) to control a grid of traffic lights for optimized traffic flow through the system. The traffic enviornment is implemented in the realistic traffic simulation [SUMO](https://sumo.dlr.de/docs/index.html). Multi-agent RL (MARL) is implemented as well with each traffic light acting as a single agent. 

## SUMO / Traci
SUMO (**S**imulation of **U**rban **MO**bility) is a continuous road traffic simulation. TraCI (**Tra**a\ffic **C**ontrol **I**nterface) connects to a SUMO simulation in a programming language (in this case Python) to allow for feeding inputs and recieving outputs. 

![SUMO picture](/images/sumo.png)
