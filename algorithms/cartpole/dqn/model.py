import torch
from torch import nn
import torch.nn.functional as F


class Agent(nn.Module):

    def __init__(
            self, dim_state=4, dim_hidden=128, dim_action=2, dim_value=1):
        super().__init__()
        self.en_state = nn.Linear(dim_state, dim_hidden)
        self.en_action = nn.Linear(dim_action, dim_hidden)
        self.decode = nn.Sequential(
            nn.Linear(2*dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, 1))

    def forward(self, s, a):
        en_s = self.en_state(s)
        en_a = self.en_action(a)
        en = torch.cat([en_s, en_a], dim=1)
        en = F.relu(en)
        value = self.decode(en)
        return value
