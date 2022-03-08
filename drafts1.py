import nntplib

import torch

from GLOBALS import *

# net = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
# a = torch.unsqueeze(torch.randn(1, 7, 7), 0)

net = nn.GRU(input_size=(32 + 128), hidden_size=128)
hidden_1 = torch.zeros(1, 1, 128)
a = torch.randn(1,1, 160)
out, hidden_2 = net(a, hidden_1)
print(out)