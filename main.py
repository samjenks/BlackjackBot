import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
from packages.model import model2 as model
"""from gym.envs.registration import register

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
)"""


def encode(x,l):
    return torch.FloatTensor([x[0], x[1], int(x[2])]).view(1,3)


def OH(x, l):

    x = torch.LongTensor([[x[0]*x[1]*(int(x[2])+1)]])
    #print("OH x:", x)
    one_hot = torch.FloatTensor(1,l)
    return one_hot.zero_().scatter_(1,x,1)


# env = gym.make('FrozenLakeNotSlippery-v0')
env = gym.make('Blackjack-v0')

# Chance of random action
e = 0.1
learning_rate = 0.01
# Discount Rate
gamma = 0.99
# Training Episodes
episodes = 1000
# Max Steps per episode
steps = 20

# Initialize history memory
step_list = []
reward_list = []
loss_list = []
e_list = []
win_list = []

state_space = 32*11*2
action_space = env.action_space.n


model = model(state_space, action_space)
optimizer = optim.Adam(model.parameters(), lr=.001)
wins = 0
loses = 0
for i in trange(episodes):

    state = env.reset()
    reward_all = 0
    done = False
    s = 0
    l = 0

    round = 1

    while round <= steps:
        state = Variable(encode(state, state_space))
        Q = model(state)
        _, action = torch.max(Q, 1)
        action = action.data[0]
        new_state, reward, done, _ = env.step(action)


        Q1 = model(Variable(encode(new_state, state_space)))
        maxQ1, _ = torch.max(Q1.data, 1)
        maxQ1 = torch.FloatTensor(maxQ1)

        targetQ = Variable(Q.data, requires_grad=False)
        targetQ[0, action] = reward + torch.mul(maxQ1, gamma)

        output = model(state)
        train_loss = F.smooth_l1_loss(output, targetQ)
        l += train_loss.data[0]

        model.zero_grad()
        train_loss.backward()
        optimizer.step()

        reward_all += reward
        print("s:", round, "player:", env.player, "dealer", env.dealer,"action", action, new_state, "reward:", reward,
              "reward sum:", reward_all, "DONE:", done)

        state = new_state

        if done:
            if reward > 0:
                wins += 1
            else:
                loses += 1
            state = env.reset()
            round += 1
            #print("s:", round, "player:", env.player, "dealer", env.dealer, "action", action, new_state, "reward:", reward,
            #     "reward sum:", reward_all)

            print("\n")
    win_list.append(wins/round)
    loss_list.append(l / round)
    reward_list.append(reward_all)

print("wins:", wins, "loses:", loses)
"""
print(win_list)
print('\nSuccessful episodes: {}'.format(np.sum(np.array(reward_list)) / episodes))
print("\nloss:", loss_list)
window = int(episodes / 10)


plt.plot(pd.Series(win_list))
plt.title('Reward Moving Average ({}-episode window)'.format(window))
plt.ylabel('Reward')
plt.xlabel('Episode')


plt.plot(pd.Series(loss_list).rolling(window).mean())
plt.title('Loss Moving Average ({}-episode window)'.format(window))
plt.ylabel('Loss')
plt.xlabel('Episode')"""

plt.show()