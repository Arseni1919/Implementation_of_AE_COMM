import torch

from GLOBALS import *
from nets import *


class Agent:
    def __init__(self, agent_name, n_agents, n_actions):
        self.name = agent_name
        # NN
        self.ie = ImageEncoder()
        self.ca = CommunicationAutoencoder()
        self.me = MessageEncoder(n_agents=n_agents)
        self.pn = PolicyNetwork(n_actions=n_actions)
        self.models_dict = {
            'ie': self.ie,
            'ca': self.ca,
            'me': self.me,
            'pn': self.pn,
        }
        # HISTORY
        self.h_obs = []
        self.h_prev_messages = []
        self.h_actions = []
        self.h_rewards = []
        self.h_dones = []

    def forward(self, observation, messages):
        # speaker module
        output_ie = self.ie(torch.unsqueeze(observation, 0))
        output_ie = torch.squeeze(output_ie)
        output_encoder, output_decoder = self.ca(torch.unsqueeze(output_ie, 0))

        # listener module
        output_me = self.me(list(messages.values()))
        # output_me = torch.unsqueeze(output_me, 0)
        input_pn = torch.unsqueeze(torch.cat((output_encoder, output_me), 1), 0)
        output_pn_probs, output_value_func = self.pn(input_pn)

        # update messages
        new_message = output_decoder
        # action = random.choice(range(6))
        categorical_distribution = Categorical(output_pn_probs)
        action = categorical_distribution.sample()
        action_log_prob = categorical_distribution.log_prob(action)

        return new_message, action

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
            for model_name, model in self.models_dict.items():
                path_to_save = f'saved_model/{env_name}_{self.name}_{model_name}.pt'
                torch.save(model, path_to_save)
            print(f"Finished saving the models of {self.name}.")

    def load_models(self, env_name):
        for model_name, model in self.models_dict.items():
            path_to_load = f'saved_model/{env_name}_{self.name}_{model_name}.pt'
            self.models_dict[model_name] = torch.load(path_to_load)
        self.ie = self.models_dict['ie']
        self.ca = self.models_dict['ca']
        self.me = self.models_dict['me']
        self.pn = self.models_dict['pn']
