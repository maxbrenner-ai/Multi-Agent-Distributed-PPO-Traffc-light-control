import torch.nn as nn
import torch
import numpy as np
from utils.utils import plot_grad_flow, layer_init_filter


# Simple one layer
class ModelBody(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelBody, self).__init__()
        self.name = 'model_body'
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.model.apply(layer_init_filter)

    def forward(self, states):
        hidden_states = self.model(states)
        return hidden_states


class ActorModel(nn.Module):
    def __init__(self, hidden_size, action_size):
        super(ActorModel, self).__init__()
        self.name = 'actor'
        self.model = nn.Sequential(
            nn.Linear(hidden_size, action_size)
        )
        self.model.apply(layer_init_filter)

    def forward(self, hidden_states):
        outputs = self.model(hidden_states)
        return outputs


class CriticModel(nn.Module):
    def __init__(self, hidden_size):
        super(CriticModel, self).__init__()
        self.name = 'critic'
        self.model = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )
        self.model.apply(layer_init_filter)

    def forward(self, hidden_states):
        outputs = self.model(hidden_states)
        return outputs


class NN_Model(nn.Module):
    def __init__(self, state_size, action_size, hidden_layer_size, device):
        super(NN_Model, self).__init__()
        self.body_model = ModelBody(state_size, hidden_layer_size).to(device)
        self.actor_model = ActorModel(hidden_layer_size, action_size).to(device)
        self.critic_model = CriticModel(hidden_layer_size).to(device)

        self.models = [self.body_model, self.actor_model, self.critic_model]

    def forward(self, states, actions=None):
        hidden_states = self.body_model(states)
        v = self.critic_model(hidden_states)
        logits = self.actor_model(hidden_states)
        dist = torch.distributions.Categorical(logits=logits)
        if actions is None:
            actions = dist.sample()
        log_prob = dist.log_prob(actions).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return {'a': actions,
                'log_pi_a': log_prob,
                'ent': entropy,
                'v': v}
