import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from GLOBALS import *


class FinalGoal(gym.Env):
    def __init__(self, n_agents=3, field_side=15, max_episode=512):
        self.name = "final_goal"
        self.field_side = field_side
        self.field = np.ndarray((self.field_side, self.field_side))
        self.n_goal_tiles = 1
        self.n_obstacle_tiles = 25
        self.n_agents = n_agents
        self.obs_side = 7
        self.n_actions = 5
        self.max_episode = max_episode
        # self.max_episode = 5
        self.action_spaces = {}
        self.observation_spaces = {}
        self.positions = {}
        self.pos_to_positions = {}
        self.agents = {}
        self.agents_list = []
        self.counter = 0
        self.to_render = False
        self.fig, (self.ax, self.ax2) = None, (None, None)
        self.agent_size = self.field_side / 50

    def reset(self):
        # create field
        pos_id = 0

        # redefine
        self.positions = {}
        self.pos_to_positions = {}
        self.agents = {}
        self.agents_list = []
        self.action_spaces = {}
        self.observation_spaces = {}
        self.counter = 0

        for i_x in range(self.field_side):
            for i_y in range(self.field_side):
                position = Position(pos_id=pos_id, x=i_x, y=i_y)
                self.positions[position.name] = position
                self.pos_to_positions[(position.x, position.y)] = position
                pos_id += 1

        # create borders in the field
        for i_pos in list(self.positions.values()):
            if i_pos.x in [0, self.field_side - 1] or i_pos.y in [0, self.field_side - 1]:
                i_pos.req = -1

        chosen = list(filter(lambda x: x.req != -1, list(self.positions.values())))
        chosen = random.sample(chosen, self.n_agents + self.n_goal_tiles + self.n_obstacle_tiles)

        # obstacles
        for i_obstacle in range(self.n_obstacle_tiles):
            chosen_pos = chosen.pop()
            chosen_pos.pos_type = 'obstacle'
            chosen_pos.req = -1

        # targets
        for i_target in range(self.n_goal_tiles):
            chosen_pos = chosen.pop()
            chosen_pos.pos_type = 'target'
            chosen_pos.req = 1

        # robots
        for i_agent in range(self.n_agents):
            chosen_pos = chosen.pop()
            agent = Agent(agent_id=i_agent, x=chosen_pos.x, y=chosen_pos.y)
            self.agents[agent.name] = agent
            self.agents_list.append(agent)
            self.action_spaces[agent.name] = gym.spaces.Discrete(self.n_actions)

        # update observations
        t_observation_to_return = self._update_observations()

        # for rendering
        # if self.to_render:
        #     plt.close()

        # return observations
        return t_observation_to_return

    def step(self, actions):
        # print(f'actions: {actions}')
        observations, rewards, dones, infos = {}, {}, {}, {}
        self.counter += 1
        for i_agent_name, i_action in actions.items():
            # position + reward
            agent = self.agents[i_agent_name]
            new_pos_x, new_pos_y = agent.x, agent.y
            if i_action == 1:  # UP
                new_pos_y = agent.y + 1
            if i_action == 2:  # DOWN
                new_pos_y = agent.y - 1
            if i_action == 3:  # LEFT
                new_pos_x = agent.x - 1
            if i_action == 4:  # RIGHT
                new_pos_x = agent.x + 1
            if (new_pos_x, new_pos_y) in self.pos_to_positions:
                curr_pos_node = self.pos_to_positions[(new_pos_x, new_pos_y)]
                if curr_pos_node.req != -1:
                    agent.x, agent.y = new_pos_x, new_pos_y
                rewards[i_agent_name] = curr_pos_node.req
            else:
                rewards[i_agent_name] = -1
            # dones
            dones[i_agent_name] = self.counter == self.max_episode
        observations = self._update_observations()
        # to tensor
        rewards = {k: torch.tensor(v).float() for k, v in rewards.items()}
        dones = {k: torch.tensor(v) for k, v in dones.items()}
        return observations, rewards, dones, infos

    def _update_observations(self):
        # update observations
        for agent_name, agent in self.agents.items():
            self.observation_spaces[agent.name] = np.zeros((self.obs_side, self.obs_side))
            for obs_x in range(self.obs_side):
                for obs_y in range(self.obs_side):
                    cell_x = agent.x - 0.5 * (self.obs_side - 1) + obs_x
                    cell_y = agent.y - 0.5 * (self.obs_side - 1) + obs_y
                    if (cell_x, cell_y) in self.pos_to_positions:
                        curr_cell = self.pos_to_positions[(cell_x, cell_y)]
                        self.observation_spaces[agent.name][obs_x][obs_y] = curr_cell.req
        # to tensor
        # return copy.deepcopy(self.observation_spaces)
        t_observation_spaces = {
            agent_name: torch.unsqueeze(torch.tensor(obs).float(), 0)
            for agent_name, obs in self.observation_spaces.items()
        }
        return t_observation_spaces

    def get_agents_names(self):
        return [agent.name for agent in self.agents_list]

    def render(self, mode="human", second_graph_dict=None, alg_name='MAS Simulation'):
        if not self.to_render:
            self.to_render = True
            self.fig, (self.ax, self.ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        if self.to_render:
            # self.fig.cla()
            positions_list = list(self.positions.values())
            agents_list = list(self.agents.values())
            self.ax.clear()
            padding = 4
            self.ax.set_xlim([0 - padding, self.field_side + padding])
            self.ax.set_ylim([0 - padding, self.field_side + padding])

            # BORDERS OF FIELD
            sm_pd = 0.5
            self.ax.plot(
                [0 - sm_pd, self.field_side - 1 + sm_pd, self.field_side - 1 + sm_pd, 0 - sm_pd, 0 - sm_pd],
                [0 - sm_pd, 0 - sm_pd, self.field_side - 1 + sm_pd, self.field_side - 1 + sm_pd, 0 - sm_pd],
                marker='o', color='brown'
            )

            # TITLES
            self.ax.set_title(alg_name)

            # POSITIONS
            target_pos = list(filter(lambda x: x.req == 1, positions_list))
            obstacle_pos = list(filter(lambda x: x.req == -1, positions_list))
            self.ax.scatter(
                [pos_node.x for pos_node in target_pos],
                [pos_node.y for pos_node in target_pos],
                alpha=[1 for _ in target_pos],
                color='g', marker="s", s=40  # s=2
            )
            self.ax.scatter(
                [pos_node.x for pos_node in obstacle_pos],
                [pos_node.y for pos_node in obstacle_pos],
                alpha=[0.7 for _ in obstacle_pos],
                color='gray', marker="s", s=40  # s=2
            )

            # ROBOTS
            for robot in agents_list:
                # robot
                circle1 = plt.Circle((robot.x, robot.y), self.agent_size, color='b', alpha=0.3)
                self.ax.add_patch(circle1)
                self.ax.annotate(robot.name, (robot.x, robot.y), fontsize=5)

                # range of obs
                obs_rectangle = plt.Rectangle((robot.x - 0.5 * (self.obs_side),
                                               robot.y - 0.5 * (self.obs_side)),
                                              self.obs_side, self.obs_side, color='tab:purple', alpha=0.15)
                self.ax.add_patch(obs_rectangle)

            # AX2
            self.ax2.clear()
            agent_name = 'agent_0'
            curr_mat = np.rot90(np.round_(self.observation_spaces[agent_name], decimals=2))
            # pprint(curr_mat)
            self.ax2.imshow(curr_mat)
            # Loop over data dimensions and create text annotations.
            for i in range(len(curr_mat[0])):
                for j in range(len(curr_mat[0])):
                    text = self.ax2.text(j, i, curr_mat[i, j], ha="center", va="center", color="w")
            self.ax2.set_title(f"{agent_name} - observation")

            plt.pause(0.05)

    def close(self):
        plt.close()


class Agent:
    def __init__(self, agent_id, x=-1, y=-1, sr=5, mr=2, cred=0.5):
        self.id = agent_id
        self.x, self.y = x, y
        self.sr = sr
        self.mr = mr
        self.cred = cred
        self.name = f'agent_{agent_id}'


class Position:
    """
    obstacle = -1
    target = 1
    robot = 0
    empty = 0
    # -1 - wall, 0 - neutral, 1 - robot, 2 - target
    """

    def __init__(self, pos_id, x, y, req=0):
        self.id = pos_id
        self.name = f'pos_{pos_id}'
        self.x, self.y = x, y
        self.req = req
        self.rem_req = req
        self.cov_req = 0
        self.pos_type = 'empty'


if __name__ == '__main__':
    game = FinalGoal()
    game.reset()
    for i_step in range(game.max_episode):
        game.step(actions={agent.name: game.action_spaces[agent.name].sample() for agent in list(game.agents.values())})
        game.render()

        print(f'\rstep: {i_step}', end='')

    print(game)
