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
from models.simple_timer_rule_set import SimpleTimerRuleSet


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


def check_rule_set_id(id):
    assert id in ['timer'], 'RULESET NOT IN POSSIBLE SETS!'


def rule_set_creator(id, params):
    check_rule_set_id(id)
    if id == 'timer':
        return SimpleTimerRuleSet(params)
