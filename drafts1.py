import nntplib

import torch

from GLOBALS import *

# net = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
# a = torch.unsqueeze(torch.randn(1, 7, 7), 0)

# net = nn.GRU(input_size=(32 + 128), hidden_size=128)
# hidden_1 = torch.zeros(1, 1, 128)
# a = torch.randn(1,1, 160)
# out, hidden_2 = net(a, hidden_1)
# print(out)

# m = nn.Softmax(dim=1)
# input = torch.randn(2, 3)
# output = m(input)
# print(output)

from small_ppo_implementation import ActorNet
from torch.distributions import Categorical
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
net = ActorNet(obs_size=3, n_actions=5)
state = torch.randn(1, 3)
probs = net(state)
# Note that this is equivalent to what used to be called multinomial
m = Categorical(probs)
action = m.sample()
lp = m.log_prob(action)
probs(action)

