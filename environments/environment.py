import traci
import numpy as np
from utils.net_scrape import *


# Base class
class Environment:
    def __init__(self, constants, device, agent_ID, eval_agent, net_path, vis=False):
        self.episode_C, self.ppo_C, self.rule_C, self.other_C = constants['episode_C'], constants['ppo_C'], \
                                                   constants['rule_C'], constants['other_C']
        self.device = device
        self.agent_ID = agent_ID
        self.eval_agent = eval_agent
        # For sumo connection
        self.conn_label = 'label_' + str(self.agent_ID)
        self.net_path = net_path
        self.vis = vis
        self.phases = None
        self.agent_type = self.other_C['agent_type']
        self.single_agent = self.other_C['single_agent']
        self.intersections = get_intersections(net_path)
        # for adding to state
        self.intersections_index = {intersection: i for i, intersection in enumerate(self.intersections)}

    def _make_state(self):
        if self.agent_type == 'rule': return {}
        if self.single_agent: return []
        else: return [[] for _ in range(len(self.intersections))]

    def _add_to_state(self, state, value, key, intersection):
        if self.agent_type == 'rule':
            if intersection:
                if intersection not in state: state[intersection] = {}
                state[intersection][key] = value
            else:
                state[key] = value  # global props for
        else:
            if self.single_agent:
                if isinstance(value, list):
                    state.extend(value)
                else:
                    state.append(value)
            else:
                if isinstance(value, list):
                    state[self.intersections_index[intersection]].extend(value)
                else:
                    state[self.intersections_index[intersection]].append(value)

    def _process_state(self, state):
        if not self.agent_type == 'rule':
            if self.single_agent:
                return np.expand_dims(np.array(state), axis=0)
            else:
                return np.array(state)
        return state

    def _open_connection(self):
        raise NotImplementedError

    def _close_connection(self):
        self.connection.close()
        del traci.main._connections[self.conn_label]

    def reset(self):
        # If there is a conn to close, then close it
        if self.conn_label in traci.main._connections:
            self._close_connection()
        # Start a new one
        self._open_connection()
        return self.get_state()

    # if action is a single number then convert to array of binary
    # and convert to dict
    # assume a is numpy array
    def _process_action(self, a):
        action = a.copy()
        if self.single_agent and self.agent_type != 'rule':
            action = '{0:0b}'.format(a[0])
            action = action.zfill(len(self.intersections))
            action = [int(c) for c in action]
        return {intersection: action[i] for i, intersection in enumerate(self.intersections)}

    def step(self, a, ep_step, get_global_reward, def_agent=False):
        action = self._process_action(a)
        if not def_agent:
            self._execute_action(action)
        self.connection.simulationStep()
        s_ = self.get_state()
        r = self.get_reward(get_global_reward)
        # Check if done (and if so reset)
        done = False
        if self.connection.simulation.getMinExpectedNumber() <= 0 or ep_step >= self.episode_C['max_ep_steps']:
            # Just close the conn without restarting if eval agent
            if self.eval_agent:
                self._close_connection()
            else:
                s_ = self.reset()
            done = True
        return s_, r, done

    def get_state(self):
        raise NotImplementedError

    def get_reward(self, get_global):
        raise NotImplementedError

    def _execute_action(self, action):
        raise NotImplementedError

    def _generate_configfile(self):
        raise NotImplementedError

    def _generate_routefile(self):
        raise NotImplementedError

    def _generate_addfile(self):
        raise NotImplementedError

