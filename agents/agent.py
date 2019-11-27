import xml.etree.ElementTree as ET
import numpy as np


# Parent - abstract
class Agent:
    def __init__(self, args, env, id):
        self.episode_C, self.model_C, self.agent_C, self.other_C, self.device = args
        self.env = env
        self.id = id

    def _get_prediction(self, states, actions=None):
        raise NotImplementedError

    def _get_action(self, prediction):
        raise NotImplementedError

    def _copy_shared_model_to_local(self):
        raise NotImplementedError

    # This method gets the ep results i really care about
    def _read_edge_waiting_time(self, file, collectEdgeIDs):
        tree = ET.parse(file)
        root = tree.getroot()
        waitingTime = []
        for c in root.iter('edge'):
            if c.attrib['id'] in collectEdgeIDs:
                waitingTime.append(float(c.attrib['waitingTime']))
        return sum(waitingTime) / len(waitingTime)

    def eval_episode(self):
        ep_rew = 0
        step = 0
        state = self.env.reset()
        while True:
            prediction = self._get_prediction(state)
            action = self._get_action(prediction)
            next_state, reward, done = self.env.step(action, step)
            ep_rew += reward
            if done:
                break
            state = np.copy(next_state)
            step += 1
        # Grab waitingTime
        waiting_time = self._read_edge_waiting_time('data/edgeData_{}.out.xml'.format(self.id), ['inEast', 'inNorth', 'inSouth', 'inWest'])
        return ep_rew, waiting_time

    def eval_episodes(self):
        self._copy_shared_model_to_local()
        test_info = {}
        test_info['all_ep_rew'] = []
        test_info['all_ep_waiting_time'] = []
        for ep in range(self.episode_C['eval_num_eps']):
            ep_rew, waiting_time = self.eval_episode()
            test_info['all_ep_rew'].append(ep_rew)
            test_info['all_ep_waiting_time'].append(waiting_time)
        return np.array(test_info['all_ep_rew']).mean(), \
                np.array(test_info['all_ep_waiting_time']).mean()