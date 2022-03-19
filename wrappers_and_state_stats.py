import torch

from GLOBALS import *


class MAEnvTensorWrapper(gym.Env):
    def __init__(self, game):
        self.game = game
        self.obs_statistics = None
        self.max_episode = self.game.max_episode

    def reset(self):
        obs_dict = self.game.reset()
        if self.obs_statistics:
            obs_dict = self.obs_statistics.get_normalized(obs_dict)
        return obs_dict

    def step(self, t_actions):
        actions = {}
        for agent_name, t_action in t_actions.items():
            actions[agent_name] = t_action.item()
        return self.game.step(actions)

    def get_agents_names(self):
        return self.game.get_agents_names()

    def render(self, mode="human", second_graph_dict=None, alg_name='MAS Simulation'):
        self.game.render(mode, second_graph_dict, alg_name)

    def close(self):
        self.game.close()


class MARunningStateStat:
    """
    https://en.wikipedia.org/wiki/Moving_average
    """
    def __init__(self, state_tensor_dict):
        self.len = 1
        self.running_mean_dict = {}
        self.running_std_dict = {}
        for agent_name, state_tensor in state_tensor_dict.items():
            state_np = state_tensor.detach().squeeze().numpy()
            self.running_mean_dict[agent_name] = state_np
            self.running_std_dict[agent_name] = state_np ** 2

    def _update(self, agent_name, state_np):
        self.len += 1
        old_mean = self.running_mean_dict[agent_name].copy()
        self.running_mean_dict[agent_name][...] = old_mean + (state_np - old_mean) / self.len
        self.running_std_dict[agent_name][...] = self.running_std_dict[agent_name] + (state_np - old_mean) * (state_np - self.running_mean_dict[agent_name])

    def mean(self, agent_name):
        return self.running_mean_dict[agent_name]

    def std(self, agent_name):
        return np.sqrt(self.running_std_dict[agent_name] / (self.len - 1))

    def get_normalized(self, state_tensor_dict):
        output_state_tensor_dict = {}
        for agent_name, state_tensor in state_tensor_dict.items():
            state_np = state_tensor.detach().squeeze().numpy()
            self._update(agent_name, state_np)
            state_np = np.clip((state_np - self.mean(agent_name)) / (self.std(agent_name) + 1e-6), -10., 10.)
            output_state_tensor_dict[agent_name] = torch.unsqueeze(torch.FloatTensor(state_np), 0)
        return output_state_tensor_dict
