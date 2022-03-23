import torch

from GLOBALS import *


class AgentNet(nn.Module):
    def __init__(self, n_agents, n_actions):
        super(AgentNet, self).__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.image_encoder = nn.Sequential(
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
        # communication_autoencoder
        self.ae_encoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.ReLU()
        )
        self.ae_decoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        # message_encoder
        self.me_embedding = nn.Linear(10, 32)
        self.message_encoder = nn.Sequential(
            nn.Linear(32 * self.n_agents, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
        # policy_network
        self.pn_gru = nn.GRU(input_size=(32 + 128), hidden_size=128)
        self.pn_hidden = torch.zeros(1, 1, 128)
        self.pn_linear_1 = nn.Linear(128, self.n_actions)
        self.pn_linear_2 = nn.Linear(128, 1)
        self.pn_relu = nn.ReLU()
        self.pn_softmax = nn.Softmax(dim=2)

    def forward(self, obs, messages: list):

        # ImageEncoder
        output_ie = self.image_encoder(obs.float())

        # CommunicationAutoencoder
        output_ie = torch.reshape(output_ie, (output_ie.shape[0], output_ie.shape[1]))
        output_ae_encoder = self.ae_encoder(output_ie.float())
        new_message = output_ae_encoder
        output_ae_decoder = self.ae_decoder(output_ae_encoder)

        # MessageEncoder
        input_me_cat = [self.me_embedding(item) for item in messages]
        input_me = torch.cat(input_me_cat, 1)
        output_me = self.message_encoder(input_me.float())
        # output_me = torch.squeeze(output_me)

        # PolicyNetwork
        # output_ie = torch.unsqueeze(output_ie, 0)
        input_pn = torch.cat((output_ie, output_me), 1)
        input_pn = torch.unsqueeze(input_pn, 1)
        output_pn_gru, self.pn_hidden = self.pn_gru(input_pn, self.pn_hidden)
        output_pn_action_logits = self.pn_linear_1(self.pn_relu(output_pn_gru))
        action_probs = self.pn_softmax(output_pn_action_logits)
        output_value_func = self.pn_linear_2(self.pn_relu(output_pn_gru))
        categorical_distribution = Categorical(action_probs)
        action = categorical_distribution.sample()
        # action_log_prob = categorical_distribution.log_prob(action)

        return action, new_message, output_ie, output_ae_decoder, action_probs, output_value_func


# class ImageEncoder(nn.Module):
#     def __init__(self):
#         super(ImageEncoder, self).__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
#             nn.ELU(),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
#             nn.ELU(),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
#             nn.ELU(),
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
#             nn.ELU(),
#             nn.AdaptiveAvgPool2d(output_size=1)
#         )
#
#     def forward(self, x):
#         return self.net(x.float())
#
#
# class CommunicationAutoencoder(nn.Module):
#     def __init__(self):
#         super(CommunicationAutoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(32, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(32, 32),
#             nn.ReLU(),
#             nn.Linear(32, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.ReLU()
#         )
#         self.final_output = nn.Linear(128, 10)
#
#     def forward(self, x):
#         encoder_output = self.encoder(x.float())
#         decoder_output = self.final_output(self.decoder(encoder_output))
#         return encoder_output, decoder_output
#
#
# class MessageEncoder(nn.Module):
#     def __init__(self, n_agents):
#         super(MessageEncoder, self).__init__()
#         self.n_agents = n_agents
#         self.embedding = nn.Linear(10, 32)
#         self.net = nn.Sequential(
#             nn.Linear(32 * self.n_agents, 32),
#             nn.ReLU(),
#             nn.Linear(32, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#         )
#
#     def forward(self, x):
#         x = torch.cat([self.embedding(item) for item in x], 1)
#         return self.net(x.float())
#
#
# class PolicyNetwork(nn.Module):
#     def __init__(self, n_actions):
#         super(PolicyNetwork, self).__init__()
#         self.n_actions = n_actions
#         self.gru = nn.GRU(input_size=(32 + 128), hidden_size=128)
#         self.linear_1 = nn.Linear(128, self.n_actions)
#         self.linear_2 = nn.Linear(128, 1)
#         self.relu = nn.ReLU()
#         self.hidden_1 = torch.zeros(1, 1, 128)
#         self.softmax_head = nn.Softmax(dim=2)
#
#     def forward(self, x):
#         output_gru, self.hidden_1 = self.gru(x, self.hidden_1)
#         output_linear = self.linear_1(self.relu(output_gru))
#         output_softmax = self.softmax_head(output_linear)
#         output_value_func = self.linear_2(self.relu(output_gru))
#         return output_softmax, output_value_func
#







