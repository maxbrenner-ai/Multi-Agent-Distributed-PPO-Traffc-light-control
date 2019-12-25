import traci
import numpy as np


# Base class
class Environment:
    def __init__(self, constants, device, agent_ID, eval_agent, vis=False):
        self.episode_C, self.model_C, self.agent_C, self.other_C = constants['episode_C'], constants['model_C'], \
                                                   constants['agent_C'], constants['other_C']
        self.device = device
        self.agent_ID = agent_ID
        self.eval_agent = eval_agent
        # For sumo connection
        self.conn_label = 'label_' + str(self.agent_ID)
        self.vis = vis
        # For returning either a dict. state (for ruleset agent) or numpy array state
        self.using_ruleset = True if self.other_C['rule_set'] else False
        self.phases = None

    def _make_state(self):
        if self.using_ruleset: return {}
        return []

    def _add_to_state(self, state, value, key, intersection):
        if self.using_ruleset:
            if intersection:
                if intersection not in state: state[intersection] = {}
                state[intersection][key] = value
            else:
                state[key] = value
        else:
            if isinstance(value, list):
                state.extend(value)
            else:
                state.append(value)

    def _process_state(self, state):
        if not self.using_ruleset: return np.expand_dims(np.array(state), axis=0)
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

    def step(self, a, ep_step, def_agent=False):
        if not def_agent:
            self._execute_action(a)
        self.connection.simulationStep()
        s_ = self.get_state()
        r = self.get_reward()
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

    def get_reward(self):
        raise NotImplementedError

    def _execute_action(self, action):
        raise NotImplementedError

    def _generate_configfile(self):
        raise NotImplementedError

    def _generate_routefile(self):
        raise NotImplementedError

    def _generate_addfile(self):
        raise NotImplementedError

