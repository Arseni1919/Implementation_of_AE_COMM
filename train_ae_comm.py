import random

from GLOBALS import *
from env_final_goal import FinalGoal
from ae_comm_agent import Agent
from neptune_plotter import NeptunePlotter
from wrappers_and_state_stats import MARunningStateStat, MAEnvTensorWrapper


def sample_trajectories(game, agents, plotter, i_update=0, batch_size=1000, to_render=True):
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
            messages_list = from_dict_to_list(agents, messages)

            # DO THE STEP
            for agent in agents:
                action, new_message = agent.forward(observations[agent.name], messages_list)
                new_messages[agent.name] = new_message
                actions[agent.name] = action
            new_observations, rewards, dones, infos = game.step(actions)

            # SAVE HISTORY
            for agent in agents:
                agent.save_history(obs=observations[agent.name], prev_m=messages_list,
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
    print(
        f'\r[SAMPLE {i_update + 1}] - episodes {n_episodes}, batch: {len(agents[0].h_rewards)}  average score: {average_score} {episode_scores}')

    return average_score


def from_dict_to_list(curr_list, curr_dict):
    return_list = []
    for li in curr_list:
        return_list.append(curr_dict[li.name])
    return return_list


def train(game, agents, plotter):
    print('Training...')
    best_score = -100
    # --------------------------- # MAIN LOOP # -------------------------- #
    for i_update in range(N_UPDATES):
        print(f'Update {i_update + 1}', end=' ')

        # SAMPLE TRAJECTORIES
        average_score = sample_trajectories(game, agents, plotter, i_update,
                                            batch_size=BATCH_SIZE, to_render=False)

        # UPDATE NN
        # TODO
        for agent in agents:
            agent.update_nn()

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
        # if i_update > N_UPDATES - 5:
        if i_update % 10 == 0:
            sample_runs(game, agents, times=1)

        # SAVE
        if average_score > best_score:
            best_score = average_score
            for agent in agents:
                agent.save_models(env_name=game.name, save_results=SAVE_RESULTS)

    # FINISH TRAINING
    print('Finished train.')


def sample_runs(game, agents, times=1, load_models=False, to_render=True):
    if load_models:
        for agent in agents:
            agent.load_models(env_name=game.name)
    # EPISODE - FULL GAME
    for i_episode in range(times):
        observations = game.reset()
        episode_score = 0
        # initial messages
        messages = {agent.name: torch.zeros(1, 10) for agent in agents}

        # STEP - ONE STEP INSIDE A GAME
        for i_step in range(game.max_episode):
            new_messages, actions = {}, {}

            # EACH AGENT DO SOME CALCULATIONS DURING THE STEP
            for agent in agents:
                messages = from_dict_to_list(agents, messages)
                action, new_message = agent.forward(observations[agent.name], messages)
                new_messages[agent.name] = new_message
                actions[agent.name] = action

            # execute actions
            new_observations, rewards, dones, infos = game.step(actions)

            # update variables
            observations = new_observations
            messages = {agent.name: new_messages[agent.name].detach() for agent in agents}
            episode_score += torch.sum(torch.tensor(list(rewards.values()))).item()

            # rendering + neptune + print
            if to_render:
                game.render()
            print(
                f'\r[RUN ({i_episode + 1}/{times})] - step: {i_step}/{game.max_episode}, episode_score: {episode_score}', end='')
        print()


def main():
    # VARIABLES
    game = FinalGoal(n_agents=N_AGENTS, field_side=FIELD_SIZE, max_episode=MAX_EPISODE_LENGTH)
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
    plotter = NeptunePlotter(plot_neptune=NEPTUNE, tags=['ae_comm', 'ppo', f'{game.name}'], name='ae_comm_sample')

    # TRAINING
    train(game=game, agents=agents_list, plotter=plotter)

    # EXAMPLE RUNS
    sample_runs(game=game, agents=agents_list, times=2, load_models=True)

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
    MAX_EPISODE_LENGTH = 50
    N_UPDATES = 100
    LR = 0.0001
    # NEPTUNE = True
    NEPTUNE = False
    # SAVE_RESULTS = True
    SAVE_RESULTS = False

    main()
