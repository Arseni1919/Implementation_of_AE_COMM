import torch

from GLOBALS import *
from nets import AgentNet
from functions import copy_nn_parameters


class Agent:
    def __init__(self, agent_name, n_agents, n_actions,
                 gamma_par=0.995, lambda_par=0.97, lr_critic_par=1e-3, lr_actor_par=1e-3, epsilon_par=0.05):
        self.name = agent_name
        self.gamma_par = gamma_par
        self.lambda_par = lambda_par
        self.lr_critic_par = lr_critic_par
        self.lr_actor_par = lr_actor_par
        self.epsilon_par = epsilon_par

        # NN
        self.nn = AgentNet(n_agents=n_agents, n_actions=n_actions)
        self.nn_old = AgentNet(n_agents=n_agents, n_actions=n_actions)
        copy_nn_parameters(self.nn_old, self.nn)
        self.critic_optim = torch.optim.Adam(self.nn.parameters(), lr=self.lr_critic_par)
        self.actor_optim = torch.optim.Adam(self.nn.parameters(), lr=self.lr_actor_par)

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

    def _history_to_tensors(self):
        self.h_obs = torch.cat(self.h_obs, 0)
        self.h_obs = torch.unsqueeze(self.h_obs, 1)
        # self.h_prev_messages = torch.tensor(self.h_prev_messages)
        self.h_rewards = torch.tensor(self.h_rewards)

    def _compute_returns_and_advantages(self):

        with torch.no_grad():
            action, new_message, output_ie, output_ae_decoder, action_probs, output_value_func = self.nn(
                self.h_obs, self.h_prev_messages
            )
            critic_values_tensor = output_value_func.detach().squeeze()
            critic_values_np = critic_values_tensor.numpy()

        returns = np.zeros(self.h_rewards.shape)
        deltas = np.zeros(self.h_rewards.shape)
        advantages = np.zeros(self.h_rewards.shape)

        prev_return, prev_value, prev_advantage = 0, 0, 0
        for i in reversed(range(self.h_rewards.shape[0])):
            final_state_bool = 1 - self.h_dones[i]

            returns[i] = self.h_rewards[i] + self.gamma_par * prev_return * final_state_bool
            prev_return = returns[i]

            deltas[i] = self.h_rewards[i] + self.gamma_par * prev_value * final_state_bool - critic_values_np[i]
            prev_value = critic_values_np[i]

            advantages[i] = deltas[i] + self.gamma_par * self.lambda_par * prev_advantage * final_state_bool
            prev_advantage = advantages[i]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
        advantages_tensor = torch.tensor(advantages).float()
        returns_tensor = torch.tensor(returns).float()

        return returns_tensor, advantages_tensor

    def _update_critic(self, returns_tensor):
        action, new_message, output_ie, output_ae_decoder, action_probs, output_value_func = self.nn(
            self.h_obs, self.h_prev_messages
        )

        critic_values_tensor = output_value_func.squeeze()
        loss_critic = nn.MSELoss()(critic_values_tensor, returns_tensor)
        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()
        return loss_critic

    def _update_actor(self, advantages_tensor):
        # UPDATE ACTOR
        action, new_message, output_ie, output_ae_decoder, action_probs_old, output_value_func = self.nn_old(
            self.h_obs, self.h_prev_messages
        )
        categorical_distribution_old = Categorical(action_probs_old)
        action_log_probs_old = categorical_distribution_old.log_prob(self.h_actions).detach()

        action, new_message, output_ie, output_ae_decoder, action_probs, output_value_func = self.nn(
            self.h_obs, self.h_prev_messages
        )
        categorical_distribution = Categorical(action_probs)
        action_log_probs = categorical_distribution.log_prob(self.h_actions)

        # UPDATE OLD NET
        copy_nn_parameters(self.nn_old, self.nn)

        ratio_of_probs = torch.exp(action_log_probs - action_log_probs_old)
        surrogate1 = ratio_of_probs * advantages_tensor
        surrogate2 = torch.clamp(ratio_of_probs, 1 - self.epsilon_par, 1 + self.epsilon_par) * advantages_tensor
        loss_actor = - torch.min(surrogate1, surrogate2)

        # ADD ENTROPY TERM
        actor_dist_entropy = categorical_distribution.entropy().detach()
        loss_actor = torch.mean(loss_actor - 1e-2 * actor_dist_entropy)
        # loss_actor = loss_actor - 1e-2 * actor_dist_entropy

        self.actor_optim.zero_grad()
        loss_actor.backward()
        # actor_list_of_grad = [torch.max(torch.abs(param.grad)).item() for param in copied_nn.parameters()]
        torch.nn.utils.clip_grad_norm_(self.nn.parameters(), 40)
        self.actor_optim.step()

        return action_probs, loss_actor

    def _remove_history(self):
        # HISTORY
        self.h_obs = []
        self.h_prev_messages = []
        self.h_actions = []
        self.h_rewards = []
        self.h_dones = []

    def update_nn(self):
        self._history_to_tensors()
        returns_tensor, advantages_tensor = self._compute_returns_and_advantages()
        loss_critic = self._update_critic(returns_tensor)
        probs, loss_actor = self._update_actor(advantages_tensor)
        self._remove_history()
        return loss_critic, loss_actor

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

