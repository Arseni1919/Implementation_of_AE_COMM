import torch

from GLOBALS import *
from nets import AgentNet


class Agent:
    def __init__(self, agent_name, n_agents, n_actions):
        self.name = agent_name

        # NN
        self.nn = AgentNet(n_agents=n_agents, n_actions=n_actions)

        # HISTORY
        self.h_obs = []
        self.h_prev_messages = []
        self.h_actions = []
        self.h_rewards = []
        self.h_dones = []

    def forward(self, obs, messages):
        obs = torch.unsqueeze(obs, 0)
        action, new_message, output_ie, output_ae_decoder, action_probs, output_value_func = self.nn(obs, messages)
        return action, new_message

    def save_history(self, obs, prev_m, action, reward, done):
        self.h_obs.append(obs)
        self.h_prev_messages.append(prev_m)
        self.h_actions.append(action)
        self.h_rewards.append(reward)
        self.h_dones.append(done)

    def _compute_returns_and_advantages(self):
        pass

    def _update_critic(self):
        pass

    def _update_actor(self):
        pass

    def _remove_history(self):
        # HISTORY
        self.h_obs = []
        self.h_prev_messages = []
        self.h_actions = []
        self.h_rewards = []
        self.h_dones = []

    def update_nn(self):
        self._compute_returns_and_advantages()
        self._update_critic()
        self._update_actor()
        self._remove_history()

    def save_models(self, env_name, save_results=False):
        if save_results:
            print(f"Saving models of {self.name}...")
            path_to_save = f'saved_model/{env_name}_{self.name}_model.pt'
            torch.save(self.nn, path_to_save)
            print(f"Finished saving the models of {self.name}.")

    def load_models(self, env_name):
        path_to_load = f'saved_model/{env_name}_{self.name}_model.pt'
        self.nn = torch.load(path_to_load)


        # self.ie = ImageEncoder()
        # self.ca = CommunicationAutoencoder()
        # self.me = MessageEncoder(n_agents=n_agents)
        # self.pn = PolicyNetwork(n_actions=n_actions)
        # self.models_dict = {
        #     'ie': self.ie,
        #     'ca': self.ca,
        #     'me': self.me,
        #     'pn': self.pn,
        # }

        # speaker module
        # output_ie = self.ie(torch.unsqueeze(obs, 0))
        # output_ie = torch.squeeze(output_ie)
        # output_encoder, output_decoder = self.ca(torch.unsqueeze(output_ie, 0))
        # new_message = output_decoder
        #
        # # listener module
        # output_me = self.me(list(messages.values()))
        # # output_me = torch.unsqueeze(output_me, 0)
        # input_pn = torch.unsqueeze(torch.cat((output_encoder, output_me), 1), 0)
        # output_pn_probs, output_value_func = self.pn(input_pn)
        #
        # # update messages
        # # action = random.choice(range(6))
        # categorical_distribution = Categorical(output_pn_probs)
        # action = categorical_distribution.sample()
        # # action_log_prob = categorical_distribution.log_prob(action)