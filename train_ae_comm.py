from GLOBALS import *
from env_final_goal import FinalGoal
from nets import *


def train():
    for i_episode in range(EPISODES):
        observations = game.reset()
        # initial messages
        messages = {agent_name: torch.zeros(1, 10) for agent_name in agents_names}

        for i_step in range(game.max_episode):
            new_messages, actions = {}, {}
            for agent_name in agents_names:

                # speaker module
                output_ie = ie_dict[agent_name](torch.unsqueeze(observations[agent_name], 0))
                output_ie = torch.squeeze(output_ie)
                output_encoder, output_decoder = ca_dict[agent_name](torch.unsqueeze(output_ie, 0))

                # listener module
                output_me = me_dict[agent_name](list(messages.values()))
                # output_me = torch.unsqueeze(output_me, 0)
                input_pn = torch.unsqueeze(torch.cat((output_encoder, output_me), 1), 0)
                output_pn = pn_dict[agent_name](input_pn)

                # update messages
                new_messages[agent_name] = output_decoder
                actions[agent_name] = game.action_spaces[agent_name].sample()

            # execute actions
            # actions = {agent.name: game.action_spaces[agent.name].sample() for agent in list(game.agents.values())}
            new_observations, rewards, dones, infos = game.step(actions)

            # update NN
            for agent_name in agents_names:
                # TODO
                pass

            # update variables
            observations = new_observations
            messages = {agent_name: new_messages[agent_name].detach() for agent_name in agents_names}

            # save the model
            # TODO

            # rendering + neptune + print
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
    # HYPER-PARAMETERS
    N_AGENTS = 3
    EPISODES = 100
    LR = 0.0001

    # VARIABLES
    game = FinalGoal(n_agents=N_AGENTS)
    game.reset()
    agents_names = game.get_agents_names()

    # NETS
    ie_dict, ca_dict, me_dict, pn_dict = {}, {}, {}, {}
    for i_a_name in agents_names:
        ie_dict[i_a_name] = ImageEncoder()
        ca_dict[i_a_name] = CommunicationAutoencoder()
        me_dict[i_a_name] = MessageEncoder(n_agents=N_AGENTS)
        pn_dict[i_a_name] = PolicyNetwork(n_actions=5)

    # TRAINING
    train()

    # EXAMPLE RUNS
    example_runs(times=2)
