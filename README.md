# TrafficControlDPPO
A work in progress. Currently, single-agent distributed PPO is used for traffic light control. Current envs are single intersection,
and multi-intersection. Very simple so far. Can run normally, grid-search or random-search from `main.py`. Just change the constants.
Right now, I am researching multi-agent RL (MARL), and then making a DPPO MARL algorithm for multi-agent traffic light control.
The multi-traffic light control env I have right now is just a single agent getting all state info from each traffic light and
then deciding whether or not to switch to the next phase for all traffic lights. MARL is a great way to reduce this dimensionality
issue. So the next add-on will be MARL for DPPO. Then comparing MARL vs single-agent for this problem. There are some papers that
use MARL for traffic light control but they mainly use DQN based algs.
