import time
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from utils import Storage, tensor, random_sample, ensure_shared_grads
import xml.etree.ElementTree as ET


#Todo: Possibly test if requires_grad=True is necc. for states

# Code adapted from: Shangtong Zhang (https://github.com/ShangtongZhang)
class PPOAgent:
    def __init__(self, args, env, shared_NN, local_NN, optimizer, id):
        self.episode_C, self.model_C, self.agent_C, self.other_C, self.device = args
        self.NN = local_NN
        self.shared_NN = shared_NN
        self.env = env
        self.state = self.env.reset()
        self.ep_step = 0
        self.opt = optimizer
        self.id = id
        self.NN.eval()

    def _get_prediction(self, states, actions=None):
        return self.NN(tensor(states, self.device), actions)

    # This method gets the ep results i really care about
    def _read_edge_waiting_time(self, file, collectEdgeIDs):
        tree = ET.parse(file)
        root = tree.getroot()
        waitingTime = []
        for c in root.iter('edge'):
            if c.attrib['id'] in collectEdgeIDs:
                waitingTime.append(float(c.attrib['waitingTime']))
        return sum(waitingTime) / len(waitingTime)

    def _eval_episode(self, test_step):
        ep_rew = 0
        step = 0
        state = self.env.reset()
        while True:
            prediction = self._get_prediction(state)
            action = prediction['a'].cpu().numpy()[0]
            next_state, reward, done = self.env.step(action, step)
            ep_rew += reward
            test_step += 1
            if done:
                break
            state = np.copy(next_state)
            step += 1
        # Grab waitingTime
        # waiting_time = 0
        waiting_time = self._read_edge_waiting_time('data/edgeData_{}.out.xml'.format(self.id), ['inEast', 'inNorth', 'inSouth', 'inWest'])
        return test_step, ep_rew, waiting_time

    def eval_episodes(self):
        self.NN.load_state_dict(self.shared_NN.state_dict())
        test_step = 0
        test_info = {}
        test_info['all_ep_rew'] = []
        test_info['all_ep_waiting_time'] = []
        for ep in range(self.episode_C['eval_num_eps']):
            test_step, ep_rew, waiting_time = self._eval_episode(test_step)
            test_info['all_ep_rew'].append(ep_rew)
            test_info['all_ep_waiting_time'].append(waiting_time)
        return np.array(test_info['all_ep_rew']).mean(), \
                np.array(test_info['all_ep_waiting_time']).mean()

    def train_rollout(self, total_step):
        storage = Storage(self.episode_C['rollout_length'])
        state = np.copy(self.state)
        step_times = []
        # Sync.
        self.NN.load_state_dict(self.shared_NN.state_dict())
        for rollout_step in range(self.episode_C['rollout_length']):
            start_step_time = time.time()
            prediction = self._get_prediction(state)
            action = prediction['a'].cpu().numpy()[0]
            next_state, reward, done = self.env.step(action, self.ep_step)

            self.ep_step += 1
            if done:
                # Sync local model with shared model at start of each ep
                self.NN.load_state_dict(self.shared_NN.state_dict())
                self.ep_step = 0

            storage.add(prediction)
            storage.add({'r': tensor(reward, self.device).unsqueeze(-1).unsqueeze(-1),
                         'm': tensor(1 - done, self.device).unsqueeze(-1).unsqueeze(-1),
                         's': tensor(state, self.device)})

            state = np.copy(next_state)

            total_step += 1

            end_step_time = time.time()
            step_times.append(end_step_time - start_step_time)

        self.state = np.copy(state)

        prediction = self._get_prediction(state)
        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((1, 1)), self.device)
        returns = prediction['v'].detach()
        for i in reversed(range(self.episode_C['rollout_length'])):
            # Disc. Return
            returns = storage.r[i] + self.agent_C['discount'] * storage.m[i] * returns
            # GAE
            td_error = storage.r[i] + self.agent_C['discount'] * storage.m[i] * storage.v[i + 1] - storage.v[i]
            advantages = advantages * self.agent_C['gae_tau'] * self.agent_C['discount'] * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        # print(returns.shape, td_error.shape, advantages.shape, storage.adv[-1].shape, storage.ret[-1].shape)

        states, actions, log_probs_old, returns, advantages = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv'])

        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        # Train
        self.NN.train()
        batch_times = []
        train_pred_times = []
        for _ in range(self.agent_C['optimization_epochs']):
            # Sync. at start of each epoch
            self.NN.load_state_dict(self.shared_NN.state_dict())
            sampler = random_sample(np.arange(states.size(0)), self.agent_C['minibatch_size'])
            for batch_indices in sampler:
                start_batch_time = time.time()

                batch_indices = tensor(batch_indices, self.device).long()

                # Important Node: these are tensors but dont have a grad
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                start_pred_time = time.time()
                prediction = self._get_prediction(sampled_states, sampled_actions)
                end_pred_time = time.time()
                train_pred_times.append(end_pred_time - start_pred_time)

                # Calc. Loss
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()

                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.agent_C['ppo_ratio_clip'],
                                          1.0 + self.agent_C['ppo_ratio_clip']) * sampled_advantages

                # policy loss and value loss are scalars
                policy_loss = -torch.min(obj, obj_clipped).mean() - self.agent_C['entropy_weight'] * prediction['ent'].mean()

                value_loss = self.agent_C['value_loss_coef'] * (sampled_returns - prediction['v']).pow(2).mean()

                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                if self.agent_C['clip_grads']:
                    nn.utils.clip_grad_norm_(self.NN.parameters(), self.agent_C['gradient_clip'])
                ensure_shared_grads(self.NN, self.shared_NN)
                self.opt.step()
                end_batch_time = time.time()
                batch_times.append(end_batch_time - start_batch_time)
        self.NN.eval()
        return total_step;''', np.array(step_times).mean(), np.array(batch_times).mean(), np.array(train_pred_times).mean()'''
