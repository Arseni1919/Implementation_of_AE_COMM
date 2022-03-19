import random

from GLOBALS import *
from env_final_goal import FinalGoal
from nets import *
from neptune_plotter import NeptunePlotter


class Agent:
    def __init__(self, agent_name, n_agents, n_actions):
        self.name = agent_name
        self.ie = ImageEncoder()
        self.ca = CommunicationAutoencoder()
        self.me = MessageEncoder(n_agents=n_agents)
        self.pn = PolicyNetwork(n_actions=n_actions)

    def forward(self, observation, messages):
        # speaker module
        output_ie = self.ie(torch.unsqueeze(observation, 0))
        output_ie = torch.squeeze(output_ie)
        output_encoder, output_decoder = self.ca(torch.unsqueeze(output_ie, 0))

        # listener module
        output_me = self.me(list(messages.values()))
        # output_me = torch.unsqueeze(output_me, 0)
        input_pn = torch.unsqueeze(torch.cat((output_encoder, output_me), 1), 0)
        output_pn_probs = self.pn(input_pn)

        # update messages
        new_message = output_decoder
        # action = random.choice(range(6))
        categorical_distribution = Categorical(output_pn_probs)
        action = categorical_distribution.sample()
        action_log_prob = categorical_distribution.log_prob(action)

        return new_message, action


def save_results(models_to_save: dict):
    if SAVE_RESULTS:
        # SAVING...
        print(f"Saving models...")
        for model_name, model in models_to_save.items():
            path_to_save = f'saved_model/{ENV_NAME}_{model_name}.pt'
            torch.save(model, path_to_save)
        print(f"Finished saving the models.")


def sample_trajectories(game, agents, batch_size=1000):
    states = {agent.name: [] for agent in agents}
    actions = {agent.name: [] for agent in agents}
    rewards = {agent.name: [] for agent in agents}
    dones = {agent.name: [] for agent in agents}
    next_states = {agent.name: [] for agent in agents}

    n_episodes = 0
    episode_scores = []

    while not len(rewards[agents[0].name]) > batch_size:
        observations = game.reset()

        # initial messages
        messages = {agent.name: torch.zeros(1, 10) for agent in agents}

        # STEP - ONE STEP INSIDE A GAME
        for i_step in range(game.max_episode):
            new_messages, actions = {}, {}

            # EACH AGENT DO SOME CALCULATIONS DURING THE STEP
            for agent in agents:

                new_message, action = agent.forward(observations[agent.name], messages)
                new_messages[agent.name] = new_message
                actions[agent.name] = action

            # execute actions
            new_observations, rewards, dones, infos = game.step(actions)

            # update variables
            observations = new_observations
            messages = {agent.name: new_messages[agent.name].detach() for agent in agents}

            # rendering + neptune + print
            game.render()
            print(f'\r(episodes {n_episodes}, steps {len(rewards)}), average score: {np.mean(episode_scores)} {episode_scores}')

    average_score = 0
    return average_score


def train(game, agents, plotter):
    print('Training...')
    best_score = -100
    # --------------------------- # MAIN LOOP # -------------------------- #
    for i_update in range(N_UPDATES):
        print(f'Update {i_update + 1}')

        # SAMPLE TRAJECTORIES
        # TODO
        average_score = sample_trajectories(game, agents, batch_size=BATCH_SIZE)

        # UPDATE NN
        # TODO
        for agent in agents:
            # COMPUTE RETURNS AND ADVANTAGES
            pass
            # UPDATE CRITIC
            pass
            # UPDATE ACTOR

        # PLOTTER
        # TODO
        plotter.neptune_plot({
            # 'critic loss': loss_critic.item(),
            # 'actor loss': loss_actor.item(),
            # 'entropy in props': Categorical(probs).entropy().mean().item(),
            # 'obs. stats - mean': obs_stat.mean().mean(),
            # 'obs. stats - std': obs_stat.std().mean(),
        })

        # RENDER
        # TODO
        # if i_update > N_UPDATES - 5:
        if i_update % 5 == 0:
            sample_runs(game, agents, times=1)

        # SAVE
        # TODO
        if average_score > best_score:
            best_score = average_score
            save_results(models_to_save=None)

    # FINISH TRAINING
    print('Finished train.')


def sample_runs(game, agents, models=None, times=1):
    # EPISODE - FULL GAME
    for i_episode in range(times):
        observations = game.reset()

        # initial messages
        messages = {agent.name: torch.zeros(1, 10) for agent in agents}

        # STEP - ONE STEP INSIDE A GAME
        for i_step in range(game.max_episode):
            new_messages, actions = {}, {}

            # EACH AGENT DO SOME CALCULATIONS DURING THE STEP
            for agent in agents:
                new_message, action = agent.forward(observations[agent.name], messages)
                new_messages[agent.name] = new_message
                actions[agent.name] = action

                # execute actions
            new_observations, rewards, dones, infos = game.step(actions)

            # update variables
            observations = new_observations
            messages = {agent.name: new_messages[agent.name].detach() for agent in agents}

            # rendering + neptune + print
            game.render()
            print(f'\repisode: {i_episode}/{EPISODES}, step: {i_step}/{game.max_episode}, loss:', end='')


def main():
    # VARIABLES
    game = FinalGoal(n_agents=N_AGENTS, field_side=FIELD_SIZE)
    game.reset()
    agents_list = [
        Agent(agent_name, n_agents=N_AGENTS, n_actions=5)
        for agent_name in game.get_agents_names()
    ]
    plotter = NeptunePlotter(plot_neptune=NEPTUNE, tags=['ae_comm', 'ppo', f'{ENV_NAME}'], name='ae_comm_sample')

    # TRAINING
    train(game=game, agents=agents_list, plotter=plotter)

    # EXAMPLE RUNS
    sample_runs(game=game, agents=agents_list, times=2)

    # FINISH
    plotter.close()
    game.close()


if __name__ == '__main__':
    # HYPER-PARAMETERS
    # N_AGENTS = 1
    BATCH_SIZE = 5000
    N_AGENTS = 3  # !!!
    FIELD_SIZE = 15  # !!!
    # FIELD_SIZE = 25
    EPISODES = 100
    N_UPDATES = 100
    LR = 0.0001
    ENV_NAME = "final_goal"
    # NEPTUNE = True
    NEPTUNE = False
    SAVE_RESULTS = True

    main()



