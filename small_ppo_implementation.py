import gym
import torch

from neptune_plotter import NeptunePlotter

from GLOBALS import *


class ActorNet(nn.Module):
    """
    obs_size: observation/state size of the environment
    n_actions: number of discrete actions available in the environment
    # hidden_size: size of hidden layers
    """

    def __init__(self, obs_size: int, n_actions: int):
        super(ActorNet, self).__init__()

        self.body_net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
        )
        self.linear_head = nn.Linear(64, n_actions)
        self.softmax_head = nn.Softmax()
        self.n_actions = n_actions
        self.obs_size = obs_size
        self.entropy_term = 0

    def forward(self, state):
        state = state.float()
        value = self.body_net(state)
        output_head = self.linear_head(value)
        output_softmax = self.softmax_head(output_head)
        return output_softmax


class CriticNet(nn.Module):
    """
    obs_size: observation/state size of the environment
    n_actions: number of discrete actions available in the environment
    # hidden_size: size of hidden layers
    """

    def __init__(self, obs_size: int):
        super(CriticNet, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )
        self.obs_size = obs_size
        self.entropy_term = 0

    def forward(self, state):
        state = state.float()
        value = self.obs_net(state)
        return value


class EnvTensorWrapper(gym.Env):
    def __init__(self, env_name='MountainCar-v0'):
        self.env_name = env_name
        self.game = gym.make(self.env_name)
        self.action_space = self.game.action_space
        self.observation_space = self.game.observation_space

    def step(self, action: torch.Tensor):
        obs, reward, done, info = self.game.step(action=action.item())
        obs = torch.tensor(obs).float()
        reward = torch.tensor(reward).float()
        done = torch.tensor(done)
        return obs, reward, done, info

    def reset(self):
        obs = self.game.reset()
        return torch.tensor(obs).float()

    def render(self, mode="human"):
        self.game.render(mode=mode)

    def close(self):
        self.game.close()

    def observation_size(self):
        if isinstance(self.game.observation_space, gym.spaces.Discrete):
            return self.game.observation_space.n
        if isinstance(self.game.observation_space, gym.spaces.Box):
            return self.game.observation_space.shape[0]
        return None

    def action_size(self):
        if isinstance(self.game.action_space, gym.spaces.Discrete):
            return self.game.action_space.n
        if isinstance(self.game.action_space, gym.spaces.Box):
            return self.game.action_space.shape[0]
        return None


def get_train_action(net, observation):
    observation = torch.unsqueeze(observation, 0)
    probs = net(observation)
    categorical_distribution = Categorical(probs)
    action = categorical_distribution.sample()
    action_log_prob = categorical_distribution.log_prob(action)
    return action, action_log_prob


def get_trajectories(scores, scores_avg):
    states, actions, rewards, dones, next_states = [], [], [], [], []

    n_episodes = 0
    episode_scores = []

    while not len(rewards) > BATCH_SIZE:
        state = env.reset()

        done = False
        episode_score = 0
        while not done:
            action, action_log_prob = get_train_action(actor_old, state)
            plotter.neptune_plot({"action": action.item()})
            next_state, reward, done, info = env.step(action)

            states.append(state.detach().squeeze().numpy())
            actions.append(action.item())
            rewards.append(reward.item())
            dones.append(done.item())
            next_states.append(next_state.detach().squeeze().numpy())

            state = next_state
            # state = state_stat.get_normalized_state(state)

            episode_score += reward.item()

        episode_scores.append(episode_score)
        n_episodes += 1
        plotter.neptune_plot({"episode_score": episode_score})

    print(f'\r(episodes {n_episodes}, steps {len(rewards)}), average score: {np.mean(episode_scores)} {episode_scores}')

    scores.append(np.mean(episode_scores))
    scores_avg.append(scores_avg[-1] * 0.9 + np.mean(episode_scores) * 0.1)
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards) / n_episodes
    dones = np.array(dones)
    next_states = np.array(next_states)

    return states, actions, rewards, dones, next_states, np.mean(episode_scores)


def compute_returns_and_advantages(rewards, dones, critic_values):
    returns = np.zeros(rewards.shape)
    deltas = np.zeros(rewards.shape)
    advantages = np.zeros(rewards.shape)

    prev_return, prev_value, prev_advantage = 0, 0, 0
    for i in reversed(range(rewards.shape[0])):
        final_state_bool = 1 - dones[i]

        returns[i] = rewards[i] + GAMMA * prev_return * final_state_bool
        prev_return = returns[i]

        deltas[i] = rewards[i] + GAMMA * prev_value * final_state_bool - critic_values[i]
        prev_value = critic_values[i]

        advantages[i] = deltas[i] + GAMMA * LAMBDA * prev_advantage * final_state_bool
        prev_advantage = advantages[i]

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
    advantages_tensor = torch.tensor(advantages).float()
    returns_tensor = torch.tensor(returns).float()

    return returns_tensor, advantages_tensor


def update_critic(states_tensor, returns_tensor):
    critic_values_tensor = critic(states_tensor).squeeze()
    loss_critic = nn.MSELoss()(critic_values_tensor, returns_tensor)
    critic_optim.zero_grad()
    loss_critic.backward()
    critic_optim.step()
    return loss_critic


def update_actor(states_tensor, actions_tensor, advantages_tensor):
    # UPDATE ACTOR
    probs_old = actor_old(states_tensor)
    categorical_distribution_old = Categorical(probs_old)
    action_log_probs_old = categorical_distribution_old.log_prob(actions_tensor).detach()

    probs = actor(states_tensor)
    categorical_distribution = Categorical(probs)
    action_log_probs = categorical_distribution.log_prob(actions_tensor)

    # UPDATE OLD NET
    for target_param, param in zip(actor_old.parameters(), actor.parameters()):
        target_param.data.copy_(param.data)

    ratio_of_probs = torch.exp(action_log_probs - action_log_probs_old)
    surrogate1 = ratio_of_probs * advantages_tensor
    surrogate2 = torch.clamp(ratio_of_probs, 1 - EPSILON, 1 + EPSILON) * advantages_tensor
    loss_actor = - torch.min(surrogate1, surrogate2)

    # ADD ENTROPY TERM
    actor_dist_entropy = categorical_distribution.entropy().detach()
    loss_actor = torch.mean(loss_actor - 1e-2 * actor_dist_entropy)
    # loss_actor = loss_actor - 1e-2 * actor_dist_entropy

    actor_optim.zero_grad()
    loss_actor.backward()
    # actor_list_of_grad = [torch.max(torch.abs(param.grad)).item() for param in actor.parameters()]
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 40)
    actor_optim.step()

    return probs, loss_actor


def save_results(model_to_save, path):
    # SAVE
    if SAVE_RESULTS:
        # SAVING...
        print(f"Saving model...")
        torch.save(model_to_save, path)
    return path


def train():
    print('Training...')
    best_score = -100
    total_scores, total_avg_scores = [0], [0]
    # --------------------------- # MAIN LOOP # -------------------------- #
    for i_update in range(N_UPDATES):
        print(f'Update {i_update + 1}')

        with torch.no_grad():
            # SAMPLE TRAJECTORIES
            states, actions, rewards, dones, next_states, average_score = get_trajectories(total_scores,
                                                                                           total_avg_scores)  # , state_stat)
            states_tensor = torch.tensor(states).float()
            actions_tensor = torch.tensor(actions).float()
            critic_values_tensor = critic(states_tensor).detach().squeeze()
            critic_values = critic_values_tensor.numpy()

            # COMPUTE RETURNS AND ADVANTAGES
            returns_tensor, advantages_tensor = compute_returns_and_advantages(rewards, dones, critic_values)

        # UPDATE CRITIC
        loss_critic = update_critic(states_tensor, returns_tensor)

        # UPDATE ACTOR
        probs, loss_actor = update_actor(states_tensor, actions_tensor, advantages_tensor)

        # PLOTTER
        plotter.neptune_plot({})

        # RENDER
        # if i_update > N_UPDATES - 5:
        if i_update % 5 == 0:
            example_run(1, actor)
            # pass

        # SAVE
        if average_score > best_score:
            best_score = average_score
            save_results(actor, path_to_save)

    # ---------------------------------------------------------------- #

    # FINISH TRAINING
    plotter.close()
    env.close()
    print('Finished train.')


def example_run(times=1, model=None):
    stars = 70
    print("*" * stars)
    for i_episode in range(times):
        obs = env.reset()
        done = False
        steps = 0
        rewards = 0
        while not done:
            if not model:
                actions = env.action_space.sample()
            else:
                # actions = env.action_space.sample()
                actions, action_log_prob = get_train_action(model, obs)
                actions = actions.item()
            obs, reward, done, info = env.step(torch.tensor(actions))
            env.render()

            steps += 1
            rewards += reward
            print(f'\r[EXAMPLE RUN]: episode: {i_episode + 1}/{times}, step: {steps}, acc. reward:{rewards}', end='')
        print()
    print("*" * stars)


def main():
    train()
    example_run(3, model=actor)


if __name__ == '__main__':
    # --------------------------- # PLOTTER & ENV # -------------------------- #
    # SEEDS
    # torch.manual_seed(0)
    # random.seed(0)
    # np.random.seed(0)

    # FOR ALGORITHM
    BATCH_SIZE = 5000
    N_UPDATES = 150
    LR_CRITIC = 1e-3
    LR_ACTOR = 1e-3
    GAMMA = 0.995  # discount factor
    EPSILON = 0.05
    SIGMA = 0.4
    LAMBDA = 0.97

    # ENV_NAME = "CartPole-v1"
    # ENV_NAME = "MountainCar-v0"
    # ENV_NAME = "MountainCarContinuous-v0"
    # ENV_NAME = 'LunarLanderContinuous-v2'
    ENV_NAME = 'LunarLander-v2'
    # ENV_NAME = "BipedalWalker-v3"

    # FOR PLOTS
    SAVE_RESULTS = True
    path_to_save = f'data/actor_{ENV_NAME}.pt'
    # NEPTUNE = True
    NEPTUNE = False

    plotter = NeptunePlotter(plot_neptune=NEPTUNE, tags=['PPO'], name='PPO_small')
    env = EnvTensorWrapper(env_name=ENV_NAME)

    # --------------------------- # NETS # -------------------------- #
    critic = CriticNet(obs_size=env.observation_size())
    actor = ActorNet(obs_size=env.observation_size(), n_actions=env.action_size())
    actor_old = ActorNet(obs_size=env.observation_size(), n_actions=env.action_size())
    # --------------------------- # OPTIMIZERS # -------------------------- #
    critic_optim = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)

    # --------------------------- # REPLAY BUFFER # -------------------------- #
    # --------------------------- # NOISE # -------------------------- #
    # current_sigma = SIGMA
    # normal_distribution = Normal(torch.tensor(0.0), torch.tensor(current_sigma))

    # Simple Ornstein-Uhlenbeck Noise generator
    # ou_noise = OUNoise()

    # --------------------------- # PLOTTER INIT # -------------------------- #
    plotter.neptune_init()

    mean_list, std_list, loss_list_actor, loss_list_critic = [], [], [], []
    list_state_mean_1, list_state_std_1 = [], []
    list_state_mean_2, list_state_std_2 = [], []
    # ---------------------------------------------------------------- #
    # ---------------------------------------------------------------- #
    main()
