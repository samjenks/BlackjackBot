import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class model(nn.Module):
    def __init__(self, state_space, action_space):
        super(model, self).__init__()
        self.input = nn.Linear(state_space, 1400)
        self.h1 = nn.Linear(1400, 2800)
        self.h2 = nn.Linear(2800, 2000)
        self.h3 = nn.Linear(2000, 1000)
        self.h4 = nn.Linear(1000, 500)
        self.h5 = nn.Linear(500, 100)
        self.policy = nn.Linear(100, action_space)

        self.bn1 = nn.BatchNorm2d(12)
        self.bn2 = nn.BatchNorm2d(24)
        self.bn3 = nn.BatchNorm2d(10)
        self.ReLu = nn.ReLU()

    def forward(self, x):
        x = self.ReLu(self.input(x))
        x = self.ReLu(self.h1(x))
        x = self.ReLu(self.h2(x))
        x = self.ReLu(self.h3(x))
        x = self.ReLu(self.h4(x))
        x = self.ReLu(self.h5(x))

        return self.policy(x)


class model2(nn.Module):
    def __init__(self, state_space, action_space):
        super(model2, self).__init__()
        self.input = nn.Linear(3, 6)
        self.h1 = nn.Linear(6, 12)
        self.h2 = nn.Linear(12, 24)
        self.h3 = nn.Linear(24, 8)
        self.h4 = nn.Linear(8, 4)
        self.policy = nn.Linear(4, action_space)

        self.bn1 = nn.BatchNorm2d(12)
        self.bn2 = nn.BatchNorm2d(24)
        self.bn3 = nn.BatchNorm2d(10)
        self.ReLu = nn.ReLU()

    def forward(self, x):
        x = self.ReLu(self.input(x))
        x = self.ReLu(self.h1(x))
        x = F.dropout(self.ReLu(self.h2(x)), training=self.training)
        x = F.dropout(self.ReLu(self.h3(x)), training=self.training)
        x = self.ReLu(self.h4(x))

        return self.policy(x)

class ModelStrat_complex(nn.Module):
    def __init__(self, state_space, action_space):
        super(ModelStrat_complex, self).__init__()
        self.rnn1 = nn.GRU(input_size=3,
                           hidden_size=128,
                           num_layers=1)
        self.dense1 = nn.Linear(128, 64)
        self.dense2 = nn.Linear(64, 32)
        self.dense3 = nn.Linear(32, 10)
        self.policy = nn.Linear(10, action_space)

    def forward(self, x, hidden):
        x = x.view(1, 1, 3)
        x, hidden = self.rnn1(x, hidden)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.policy(x)
        return x.view(1,2), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return Variable(weight.new(1, batch_size, 128).zero_())
