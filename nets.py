import torch

from GLOBALS import *


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d(output_size=1)
        )

    def forward(self, x):
        return self.net(x.float())


class CommunicationAutoencoder(nn.Module):
    def __init__(self):
        super(CommunicationAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.final_output = nn.Linear(128, 10)

    def forward(self, x):
        encoder_output = self.encoder(x.float())
        decoder_output = self.final_output(self.decoder(encoder_output))
        return encoder_output, decoder_output


class MessageEncoder(nn.Module):
    def __init__(self, n_agents):
        super(MessageEncoder, self).__init__()
        self.n_agents = n_agents
        self.embedding = nn.Linear(10, 32)
        self.net = nn.Sequential(
            nn.Linear(32 * self.n_agents, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )

    def forward(self, x):
        x = torch.cat([self.embedding(item) for item in x], 1)
        return self.net(x.float())


class PolicyNetwork(nn.Module):
    def __init__(self, n_actions):
        super(PolicyNetwork, self).__init__()
        self.n_actions = n_actions
        self.gru = nn.GRU(input_size=(32 + 128), hidden_size=128)
        self.linear_1 = nn.Linear(128, self.n_actions)
        self.linear_2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.hidden_1 = torch.zeros(1, 1, 128)
        self.softmax_head = nn.Softmax(dim=2)

    def forward(self, x):
        output_gru, self.hidden_1 = self.gru(x, self.hidden_1)
        output_linear = self.linear_1(self.relu(output_gru))
        output_softmax = self.softmax_head(output_linear)
        output_value_func = self.linear_2(self.relu(output_gru))
        return output_softmax, output_value_func








