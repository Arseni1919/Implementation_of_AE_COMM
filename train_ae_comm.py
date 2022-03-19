import random

from GLOBALS import *
from env_final_goal import FinalGoal
from ae_comm_agent import Agent
from neptune_plotter import NeptunePlotter
from wrappers_and_state_stats import MARunningStateStat, MAEnvTensorWrapper


def save_results(models_to_save: dict):
    if SAVE_RESULTS:
        # SAVING...
        print(f"Saving models...")
        for model_name, model in models_to_save.items():
            path_to_save = f'saved_model/{ENV_NAME}_{model_name}.pt'
            torch.save(model, path_to_save)
        print(f"Finished saving the models.")


def sample_trajectories(game, agents, plotter, batch_size=1000, to_render=True):
    n_episodes = 0
    episode_scores = []

    # BATCH
    while not len(agents[0].h_rewards) > batch_size:
        observations = game.reset()
        episode_score = 0
        # initial messages
        messages = {agent.name: torch.zeros(1, 10) for agent in agents}

        # STEP - ONE STEP INSIDE A GAME
        for i_step in range(game.max_episode):
            print(f'\r(step: {len(agents[0].h_rewards)})', end='')
            new_messages, actions = {}, {}

            # DO THE STEP
            for agent in agents:
                new_message, action = agent.forward(observations[agent.name], messages)
                new_messages[agent.name] = new_message
                actions[agent.name] = action
            new_observations, rewards, dones, infos = game.step(actions)

            # SAVE HISTORY
            for agent in agents:
                agent.save_history(obs=observations[agent.name], prev_m=messages[agent.name],
                                   action=actions[agent.name], reward=rewards[agent.name],
                                   done=dones[agent.name])

            # update variables
            observations = new_observations
            messages = {agent.name: new_messages[agent.name].detach() for agent in agents}
            episode_score += torch.sum(torch.tensor(list(rewards.values()))).item()

            # rendering
            if to_render:
                game.render()

        # after finishing the episode
        episode_scores.append(episode_score)
        n_episodes += 1
        plotter.neptune_plot({"episode_score": episode_score})

    # after finishing the batch
    average_score = np.mean(episode_scores)
    print(f'\r[SAMPLE] - episodes {n_episodes}, batch: {len(agents[0].h_rewards)}  average score: {average_score} {episode_scores}')

    return average_score


def train(game, agents, plotter):
    print('Training...')
    best_score = -100
    # --------------------------- # MAIN LOOP # -------------------------- #
    for i_update in range(N_UPDATES):
        print(f'Update {i_update + 1}')

        # SAMPLE TRAJECTORIES
        # TODO
        average_score = sample_trajectories(game, agents, plotter,
                                            batch_size=BATCH_SIZE, to_render=False)

        # UPDATE NN
        # TODO
        for agent in agents:
            agent.update_nn()
            agent.remove_history()

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
            for agent in agents:
                agent.save_models()

    # FINISH TRAINING
    print('Finished train.')


def sample_runs(game, agents, models=None, times=1):
    # TODO: load models
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
    # --------------------------- # WRAPPERS & OBS STATS # -------------------------- #
    game = MAEnvTensorWrapper(game=game)
    obs_stat = MARunningStateStat(game.reset())
    game.obs_statistics = obs_stat
    # ------------------------------------------------------------------------------- #
    game.reset()
    agents_list = [
        Agent(agent_name, n_agents=N_AGENTS, n_actions=5)
        for agent_name in game.get_agents_names()
    ]
    plotter = NeptunePlotter(plot_neptune=NEPTUNE, tags=['ae_comm', 'ppo', f'{ENV_NAME}'], name='ae_comm_sample')

    # TRAINING
    train(game=game, agents=agents_list, plotter=plotter)

    # EXAMPLE RUNS
    # TODO: load models
    sample_runs(game=game, agents=agents_list, times=2)

    # FINISH
    plotter.close()
    game.close()


if __name__ == '__main__':
    # HYPER-PARAMETERS
    # N_AGENTS = 1
    BATCH_SIZE = 1000
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



