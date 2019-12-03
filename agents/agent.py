import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict


# Parent - abstract
class Agent:
    def __init__(self, args, env, id, data_collector):
        self.episode_C, self.model_C, self.agent_C, self.other_C, self.device = args
        self.env = env
        self.id = id
        self.data_collector = data_collector

    def _reset(self):
        raise NotImplementedError

    def _get_prediction(self, states, actions=None):
        raise NotImplementedError

    def _get_action(self, prediction):
        raise NotImplementedError

    def _copy_shared_model_to_local(self):
        raise NotImplementedError

    # This method gets the ep results i really care about (gets averages)
    def _read_edge_results(self, file, collectEdgeIDs, results):
        tree = ET.parse(file)
        root = tree.getroot()
        for c in root.iter('edge'):
            if c.attrib['id'] in collectEdgeIDs:
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
            prediction = self._get_prediction(state)
            action = self._get_action(prediction)
            next_state, reward, done = self.env.step(action, step)
            ep_rew += reward
            if done:
                break
            state = np.copy(next_state)
            step += 1
        results = self._read_edge_results('data/edgeData_{}.out.xml'.format(self.id), ['inEast', 'inNorth', 'inSouth', 'inWest'], results)
        results['rew'].append(ep_rew)
        return results

    def eval_episodes(self, current_rollout, ep_count=None):
        self._copy_shared_model_to_local()
        results = defaultdict(list)
        for ep in range(self.episode_C['eval_num_eps']):
            results = self.eval_episode(results)
        # So I could add the data from each ep to the data collector but I like the smoothing effect this has
        if current_rollout:
            results['rollout'] = [current_rollout]
        results = {k: sum(v) / len(v) for k, v in list(results.items())}
        self.data_collector.collect_ep(results, ep_count+self.episode_C['eval_num_eps'])
