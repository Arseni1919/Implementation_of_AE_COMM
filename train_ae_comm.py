from GLOBALS import *
from env_final_goal import FinalGoal
from nets import *


class Agent:
    def __init__(self, agent_name, n_agents, n_actions):
        self.name = agent_name
        self.ie = ImageEncoder()
        self.ca = CommunicationAutoencoder()
        self.me = MessageEncoder(n_agents=n_agents)
        self.pn = PolicyNetwork(n_actions=n_actions)


def train(game, agents):
    for i_episode in range(EPISODES):
        observations = game.reset()
        # initial messages
        messages = {agent.name: torch.zeros(1, 10) for agent in agents}

        for i_step in range(game.max_episode):
            new_messages, actions = {}, {}
            for agent in agents:

                # speaker module
                output_ie = agent.ie(torch.unsqueeze(observations[agent.name], 0))
                output_ie = torch.squeeze(output_ie)
                output_encoder, output_decoder = agent.ca(torch.unsqueeze(output_ie, 0))

                # listener module
                output_me = agent.me(list(messages.values()))
                # output_me = torch.unsqueeze(output_me, 0)
                input_pn = torch.unsqueeze(torch.cat((output_encoder, output_me), 1), 0)
                output_pn = agent.pn(input_pn)

                # update messages
                new_messages[agent.name] = output_decoder
                actions[agent.name] = game.action_spaces[agent.name].sample()

            # execute actions
            # actions = {agent.name: game.action_spaces[agent.name].sample() for agent in list(game.agents.values())}
            new_observations, rewards, dones, infos = game.step(actions)

            # update NN
            for agent in agents:
                # TODO
                pass

            # update variables
            observations = new_observations
            messages = {agent.name: new_messages[agent.name].detach() for agent in agents}

            # save the model
            # TODO

            # rendering + neptune + print
            game.render()
            print(f'\repisode: {i_episode}/{EPISODES}, step: {i_step}/{game.max_episode}, loss:', end='')


def example_runs(game, agents, times=1):
    pass
    # for i_episode in range(times):
    #     game.reset()
    #     for i_step in range(game.max_episode):
    #         actions = {agent.name: game.action_spaces[agent.name].sample() for agent in list(game.agents.values())}
    #         game.step(actions)
    #         game.render()
    #
    #         print(f'\repisode: {i_episode}/{EPISODES}, step: {i_step}/{game.max_episode}, loss:', end='')


def main():
    # VARIABLES
    game = FinalGoal(n_agents=N_AGENTS, field_side=FIELD_SIZE)
    game.reset()
    agents = [
        Agent(agent_name, n_agents=N_AGENTS, n_actions=5)
        for agent_name in game.get_agents_names()
    ]

    # TRAINING
    train(game=game, agents=agents)

    # EXAMPLE RUNS
    example_runs(game=game, agents=agents, times=2)


if __name__ == '__main__':
    # HYPER-PARAMETERS
    N_AGENTS = 1
    # FIELD_SIZE = 15  !!!
    FIELD_SIZE = 10
    EPISODES = 100
    LR = 0.0001

    main()



