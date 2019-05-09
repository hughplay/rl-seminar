import torch
from torch import nn
from torch.distributions import Categorical


class Agent(nn.Module):

    def __init__(self, dim_state=4, dim_hidden=128, dim_action=2, dim_value=1):
        super().__init__()
        self.share = nn.Sequential(
            nn.Linear(dim_state, dim_hidden),
            nn.ReLU())
        self.action_head = nn.Sequential(
            nn.Linear(dim_hidden, dim_action),
            nn.Softmax(dim=-1))
        self.value_head = nn.Linear(dim_hidden, dim_value)

    def probs2action(self, probs):
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action, log_prob

    def forward(self, x):
        x = self.share(x)
        action_dist = self.action_head(x)
        value = self.value_head(x)

        action, log_prob = self.probs2action(action_dist)

        return action, log_prob, value
