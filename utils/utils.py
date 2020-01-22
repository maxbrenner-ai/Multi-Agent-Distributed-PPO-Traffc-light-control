import matplotlib.pyplot as plt
from collections import OrderedDict
import pickle
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
import torch.nn as nn
import torch
import random
import torch.multiprocessing as mp
from models.timer_rule_set import TimerRuleSet
from models.cycle_rule_set import CycleRuleSet


# Acknowledge: Ilya Kostrikov (https://github.com/ikostrikov)
# This assigns this agents grads to the shared grads at the very start
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


# Acknowledge: Alexis Jacq (https://github.com/alexis-jacq)
class Counter:
    """enable the chief to access worker's total number of updates"""

    def __init__(self):
        self.val = mp.Value("i", 0)
        self.lock = mp.Lock()

    def get(self):
        # used by chief
        with self.lock:
            return self.val.value

    def increment(self, amt=1):
        # used by workers
        with self.lock:
            self.val.value += amt

    def reset(self):
        # used by chief
        with self.lock:
            self.val.value = 0


# Works for both single contants and lists for grid
def load_constants(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data


def refresh_excel(filepath, excel_header):
    df = pd.DataFrame(columns=excel_header)
    df.to_excel(filepath, index=False, header=excel_header)
    

# Acknowledge: Shangtong Zhang (https://github.com/ShangtongZhang)
class Storage:
    def __init__(self, size, keys=None):
        if keys is None:
            keys = []
        keys = keys + ['s', 'a', 'r', 'm',
                       'v', 'q', 'pi', 'log_pi', 'ent',
                       'adv', 'ret', 'q_a', 'log_pi_a',
                       'mean']
        self.keys = keys
        self.size = size
        self.reset()

    def add(self, data):
        for k, v in data.items():
            if k not in self.keys:
                self.keys.append(k)
                setattr(self, k, [])
            getattr(self, k).append(v)

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.size)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    def cat(self, keys):
        data = [getattr(self, k)[:self.size] for k in keys]
        return map(lambda x: torch.cat(x, dim=0), data)
 

# Acknowledge: Shangtong Zhang (https://github.com/ShangtongZhang)
def tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float)
    x = torch.tensor(x, device=device, dtype=torch.float32)
    return x
  

# Acknowledge: Shangtong Zhang (https://github.com/ShangtongZhang)
def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]


def layer_init_filter(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    
    
def layer_init(layer, w_scale=1.0):
    nn.init.xavier_uniform_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

    
def plot_grad_flow(layers, ave_grads, max_grads):
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


def get_rule_set_class(id):
    if id == 'timer':
        return TimerRuleSet
    if id == 'cycle':
        return CycleRuleSet
    raise AssertionError('Ruleset not in possible sets!')


# For traffic probability based on episode step (NOT USED)
class PiecewiseLinearFunction:
    def __init__(self, points):
        self.points = points
        self.linear_funcs = []
        for p in range(len(points)-1):
            if p % 2 == 1:
                continue
            self.linear_funcs.append(LinearFunction(points[p], points[p+1]))

    def get_output(self, x):
        for lf in self.linear_funcs:
            output = lf.func(x)
            if output is not None:
                return output
        # If got here something wrong with the lfs sent in
        assert True is False, 'Something wrong with lfs sent in.'

    def visualize(self):
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        plt.plot(xs, ys)
        plt.show()


class LinearFunction:
    def __init__(self, p1, p2):
        x0, y0 = p1
        x1, y1 = p2
        domain = (min(x0, x1), max(x0, x1))
        m = (y1 - y0) / (x1 - x0)
        b = y0 - m * x0
        self.func = lambda x: m * x + b if domain[0] <= x < domain[1] else None


def get_net_path(constants):
    shape = constants['environment']['shape']
    file = 'data/'
    file += '{}_{}_'.format(shape[0], shape[1])
    file += 'intersections.net.xml'
    return file


def get_state_action_size(PER_AGENT_STATE_SIZE, GLOBAL_STATE_SIZE, ACTION_SIZE, constants):
    def get_num_agents():
        shape = constants['environment']['shape']
        return int(shape[0] * shape[1])
    single_agent = constants['agent']['single_agent']
    # if single agent then make sure to scale up the state size and action size
    if single_agent:
        return {'s': int((get_num_agents() * PER_AGENT_STATE_SIZE) + GLOBAL_STATE_SIZE), 'a': int(pow(ACTION_SIZE, get_num_agents()))}
    else:
        return {'s': int(PER_AGENT_STATE_SIZE + GLOBAL_STATE_SIZE), 'a': int(ACTION_SIZE)}
