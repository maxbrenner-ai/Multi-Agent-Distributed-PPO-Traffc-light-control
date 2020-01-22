import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict
from copy import deepcopy


# Parent - abstract
class Worker:
    def __init__(self, constants, device, env, id, data_collector):
        self.constants = constants
        self.device = device
        self.env = env
        self.id = id
        self.data_collector = data_collector

    def _reset(self):
        raise NotImplementedError

    def _get_prediction(self, states, actions=None, ep_step=None):
        raise NotImplementedError

    def _get_action(self, prediction):
        raise NotImplementedError

    def _copy_shared_model_to_local(self):
        raise NotImplementedError

    # This method gets the ep results i really care about (gets averages)
    def _read_edge_results(self, file, results):
        tree = ET.parse(file)
        root = tree.getroot()
        for c in root.iter('edge'):
            for k, v in list(c.attrib.items()):
                if k == 'id': continue
                results[k].append(float(v))
        return results

    # ep_count only for test
    def eval_episode(self, results):
        ep_rew = 0
        step = 0
        state = self.env.reset()
        self._reset()
        while True:
            prediction = self._get_prediction(state, ep_step=step)
            action = self._get_action(prediction)
            next_state, reward, done = self.env.step(action, step, get_global_reward=True)
            ep_rew += reward
            if done:
                break
            if not isinstance(state, dict):
                state = np.copy(next_state)
            else:
                state = deepcopy(next_state)
            step += 1
        results = self._read_edge_results('data/edgeData_{}.out.xml'.format(self.id), results)
        results['rew'].append(ep_rew)
        return results

    def eval_episodes(self, current_rollout, model_state=None, ep_count=None):
        self._copy_shared_model_to_local()
        results = defaultdict(list)
        for ep in range(self.constants['episode']['eval_num_eps']):
            results = self.eval_episode(results)
        # So I could add the data from each ep to the data collector but I like the smoothing effect this has
        if current_rollout:
            results['rollout'] = [current_rollout]
        results = {k: sum(v) / len(v) for k, v in list(results.items())}
        self.data_collector.collect_ep(results, model_state, ep_count+self.constants['episode']['eval_num_eps'] if ep_count is not None else None)
