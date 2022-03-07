from GLOBALS import *
from env_final_goal import FinalGoal


def train():
    for i_episode in range(EPISODES):
        observations = game.reset()
        for i_step in range(game.max_episode):
            # collect messages and observations
            # TODO

            # speaker module
            # TODO

            # listener module
            # TODO

            # execute actions
            actions = {agent.name: game.action_spaces[agent.name].sample() for agent in list(game.agents.values())}
            new_observations, rewards, dones, infos = game.step(actions)

            # update NN
            # TODO

            # update variables
            new_observations = observations

            # save the model
            # TODO

            # rendering
            game.render()

            print(f'\repisode: {i_episode}/{EPISODES}, step: {i_step}/{game.max_episode}, loss:', end='')


def example_runs(times=1):
    pass
    # for i_episode in range(times):
    #     game.reset()
    #     for i_step in range(game.max_episode):
    #         actions = {agent.name: game.action_spaces[agent.name].sample() for agent in list(game.agents.values())}
    #         game.step(actions)
    #         game.render()
    #
    #         print(f'\repisode: {i_episode}/{EPISODES}, step: {i_step}/{game.max_episode}, loss:', end='')


if __name__ == '__main__':
    # PARAMETERS
    EPISODES = 100

    # VARIABLES
    game = FinalGoal()

    # TRAINING
    train()

    # EXAMPLE RUNS
    example_runs(times=2)