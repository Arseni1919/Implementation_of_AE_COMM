import math
import random
from abc import ABC

import numpy as np

from GLOBALS import *
from functions import distance_nodes





"""
Full API:
Attributes: 
    agents, num_agents, possible_agents, max_num_agents, observation_spaces, action_spaces

Methods: 
    render(mode='human'), 
    seed(seed=None), close(), 
    observation_space(agent), 
    action_space(agent)

    step(actions): receives a dictionary of actions keyed by the agent name. Returns the observation dictionary, 
                   reward dictionary, done dictionary, and info dictionary, where each dictionary is keyed by the agent.

    reset(): resets the environment and returns a dictionary of observations (keyed by the agent name)

Functions:
    random_demo(env, render=True, episodes=1)

"""


class MSASimulatorParallel:

    def __init__(self, num_agents, to_render=True, poi=10, width=50, target_radius=10,
                 agent_sr=5, agent_mr=2, agent_cred=0.5):
        self.agents = None
        self.agents_list = None
        self.num_agents = num_agents
        self.possible_agents = self.agents_list
        self.max_num_agents = num_agents
        self.observation_spaces = None
        self.action_spaces = None

        self.field_dict = None
        self.field_list = None
        self.width = width
        self.poi = None  # POINTS OF INTEREST
        self.num_of_poi = poi
        self.poi_radius = target_radius
        self.agent_sr = agent_sr
        self.agent_mr = agent_mr
        self.agent_cred = agent_cred

        # RENDER
        self.to_render = to_render
        self.agent_size = self.width / 50
        self.rewards_sum_list = None
        if self.to_render:
            # self.fig, self.ax = plt.subplots(figsize=[6.5, 6.5])
            self.fig, (self.ax, self.ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
            # self.fig, (self.ax, self.ax2, self.ax3) = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))

    @staticmethod
    def seed():
        SEED = 123
        # torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        # env.seed(SEED)

    def observation_space(self, agent):
        return

    def action_space(self, agent):
        return self.agents[agent].actions

    def _get_observations(self):
        observations = {}
        for agent in self.agents_list:
            observations[agent.name] = []
            for pos in self.field_list:
                if distance_nodes(agent, pos) <= agent.sr:
                    observations[agent.name].append((pos.x, pos.y, pos.req))
        return observations

    def reset(self):
        # CLEAR
        self.field_list, self.agents_list = [], []
        self.field_dict, self.agents = {}, {}
        self.rewards_sum_list = []

        # CREATE FIELD
        for i_x in range(self.width):
            for i_y in range(self.width):
                pos = Position(pos_id=f'{i_x}{i_y}', x=i_x, y=i_y)
                self.field_list.append(pos)
        self.field_dict = {pos.name: pos for pos in self.field_list}

        # CREATE AGENTS
        positions_for_agents = random.sample(self.field_list, self.num_agents)
        self.agents_list = [
            Agent(
                i, pos.x, pos.y, sr=self.agent_sr, mr=self.agent_mr, cred=self.agent_cred
            ) for i, pos in enumerate(positions_for_agents)
        ]
        self.agents = {agent.name: agent for agent in self.agents_list}

        # CREATE POINTS OF INTEREST
        self.poi = random.sample(self.field_list, self.num_of_poi)
        # self.poi = [self.field_list[25]]
        for target in self.poi:
            for pos in self.field_list:
                dist = distance_nodes(target, pos)
                if dist <= 1.0:
                    new_req = 1.0
                elif 1.0 < dist <= self.poi_radius:
                    new_req = min(1.0, ((1 / dist) + pos.req))
                else:
                    new_req = pos.req
                pos.req = new_req

        # BUILD FIRST OBSERVATIONS
        return self._get_observations()

    def get_next_pos(self, agent, action):
        new_pos_x, new_pos_y = agent.x, agent.y
        # if action == 1:  # UP
        #     new_pos_y = agent.y + 1
        # if action == 2:  # DOWN
        #     new_pos_y = agent.y - 1
        # if action == 3:  # LEFT
        #     new_pos_x = agent.x - 1
        # if action == 4:  # RIGHT
        #     new_pos_x = agent.x + 1
        if action == 1:  # UP
            new_pos_y = agent.y + 1
        if action == 2:  # RIGHT
            new_pos_x = agent.x + 1
            new_pos_y = agent.y + 1
        if action == 3:  # RIGHT
            new_pos_x = agent.x + 1
        if action == 4:  # RIGHT
            new_pos_x = agent.x + 1
            new_pos_y = agent.y - 1
        if action == 5:  # DOWN
            new_pos_y = agent.y - 1
        if action == 6:  # RIGHT
            new_pos_y = agent.y - 1
            new_pos_x = agent.x - 1
        if action == 7:  # LEFT
            new_pos_x = agent.x - 1
        if action == 8:  # RIGHT
            new_pos_x = agent.x - 1
            new_pos_y = agent.y + 1

        new_pos_x = min(self.width - 1, max(0, new_pos_x))
        new_pos_y = min(self.width - 1, max(0, new_pos_y))

        return new_pos_x, new_pos_y

    def step(self, actions: dict):
        for agent_name, action in actions.items():
            agent = self.agents[agent_name]
            new_pos_x, new_pos_y = self.get_next_pos(agent, action)
            agent.x = new_pos_x
            agent.y = new_pos_y

        observations = self._get_observations()
        rewards = self._get_rewards(observations)
        self.rewards_sum_list.append(sum(rewards.values()))
        dones, infos = {}, {}
        return observations, rewards, dones, infos

    @staticmethod
    def _get_rewards(observations):
        rewards = {}
        for agent_name, positions in observations.items():
            rewards[agent_name] = 0
            for pos in positions:
                rewards[agent_name] += pos[2]
        return rewards

    def random_demo(self, render=True, episodes=1):
        pass

    def render(self, second_graph_dict=None, alg_name='MAS Simulation'):
        if self.to_render:
            # self.fig.cla()

            self.ax.clear()
            padding = 4
            self.ax.set_xlim([0 - padding, self.width + padding])
            self.ax.set_ylim([0 - padding, self.width + padding])

            # BORDERS OF FIELD
            sm_pd = 0.5
            self.ax.plot(
                [0 - sm_pd, self.width - 1 + sm_pd, self.width - 1 + sm_pd, 0 - sm_pd, 0 - sm_pd],
                [0 - sm_pd, 0 - sm_pd, self.width - 1 + sm_pd, self.width - 1 + sm_pd, 0 - sm_pd],
                marker='o', color='brown'
            )

            # TITLES
            self.ax.set_title(alg_name)

            # POSITIONS
            self.ax.scatter(
                [pos_node.x for pos_node in self.field_list],
                [pos_node.y for pos_node in self.field_list],
                alpha=[pos_node.req for pos_node in self.field_list],
                color='g', marker="s", s=5  # s=2
            )

            # ROBOTS
            for robot in self.agents_list:
                # robot
                circle1 = plt.Circle((robot.x, robot.y), self.agent_size, color='b', alpha=0.3)
                self.ax.add_patch(circle1)
                self.ax.annotate(robot.name, (robot.x, robot.y), fontsize=5)

                # range of sr
                circle_sr = plt.Circle((robot.x, robot.y), robot.sr, color='y', alpha=0.15)
                self.ax.add_patch(circle_sr)

                # range of mr
                circle_mr = plt.Circle((robot.x, robot.y), robot.mr, color='tab:purple', alpha=0.15)
                self.ax.add_patch(circle_mr)

            if second_graph_dict:
                self.ax2.clear()
                self.ax2.set_title(second_graph_dict['name'])
                self.ax2.scatter(
                    [pos_node.x for pos_node in second_graph_dict['nodes']],
                    [pos_node.y for pos_node in second_graph_dict['nodes']],
                    alpha=[pos_node.req for pos_node in second_graph_dict['nodes']],
                    color='darkred', marker="p", s=100
                )

            plt.pause(0.05)

    def get_field(self, req=1):
        field = []
        for pos in self.field_list:
            field.append(Position(pos.id, pos.x, pos.y, req=req))
        return field


class Agent:
    def __init__(self, agent_id, x=-1, y=-1, sr=5, mr=2, cred=0.5):
        self.id = agent_id
        self.x, self.y = x, y
        self.sr = sr
        self.mr = mr
        self.cred = cred
        self.name = f'agent_{agent_id}'
        # self.actions = [0, 1, 2, 3, 4]
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]


class Position:
    def __init__(self, pos_id, x, y, req=0):
        self.id = pos_id
        self.name = f'pos_{pos_id}'
        self.x, self.y = x, y
        self.req = req
        self.rem_req = req
        self.cov_req = 0
