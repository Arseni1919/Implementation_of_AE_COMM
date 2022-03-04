import gym
import numpy as np

from GLOBALS import *


class FinalGoal(gym.Env):
    def __init__(self):
        self.side = 15
        self.field = np.ndarray((self.side, self.side))
        self.n_goal_tiles = 1
        self.n_obstacle_tiles = 25
        self.n_agents = 3
        self.obs_side = 7
        self.n_actions = 5
        self.max_episode = 512
        self.action_spaces = {}
        self.observation_spaces = {}

    def reset(self):
        # create action and observation spaces
        for i_agent in range(self.n_agents):
            name = f'agent_{i_agent}'
            self.action_spaces[name] = gym.spaces.Discrete(self.n_actions)
            self.observation_spaces[name] = np.ndarray((self.obs_side, self.obs_side))

        # create field
        # -1 - wall, 0 - neutral, 1 - robot, 2 - target

        # update observations

        # return observations

    def step(self, action):
        pass

    def render(self, mode="human"):
        pass


game = FinalGoal()
game.reset()
print(game)