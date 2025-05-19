# %%
import contextlib
import logging
import pathlib
import time
from collections import deque
from dataclasses import dataclass
import os
import pdb
import numpy as np
import scenic
import torch
import torch.multiprocessing as mp
import tyro
import gymnasium as gym
from gymnasium import spaces
# from scenic.gym import ScenicGymEnv
from scenic.zoo import ScenicZooEnv
from scenic.simulators.metadrive import MetaDriveSimulator
from torch import nn, optim
import pickle


device = "cuda" if torch.cuda.is_available() else "cpu"

# %%


@dataclass
class Args:
    """Hyperparameters and configuration for the PPO training."""

    # Environment/scenic file to use
    scenic_file: str = "eval.scenic"
    # Number of parallel processes for data collection
    num_workers: int = 16
    # Total timesteps for eval
    total_timesteps: int = 30_000 
    # Timesteps collected by each worker per iteration
    steps_per_worker: int = 256
    # Number of optimization epochs per PPO iteration
    num_epochs: int = 4
    # Size of minibatches for optimization
    minibatch_size: int = 6
    # Discount factor
    gamma: float = 0.9
    # Lambda for Generalized Advantage Estimation
    gae_lambda: float = 0.9
    # PPO clipping parameter
    clip_epsilon: float = 0.2
    # Learning rate
    lr: float = 3e-4
    # Entropy coefficient for exploration bonus
    entropy_coef: float = 0.01
    # Value function loss coefficient
    value_loss_coef: float = 0.5
    # Gradient clipping threshold
    max_grad_norm: float = 0.5
    # Random seed
    seed: int = 4
    # Directory to save models
    model_dir: str = "models"
    
    use_pretrained: bool = True
    
    checkpoint_model: str = "models/ce_deviation_05_18_19_43.pth"

    eval_results_folder: str = "eval_results"

    eval_result_file_name: str = "ce_deviation_05_18_19_43_eval"


LOG_STD_MAX = 2
LOG_STD_MIN = -5
EPSILON = 1e-5


class ActorCritic(nn.Module):
    """A simple Actor-Critic network for discrete action spaces. Shares layers between actor and critic."""

    def __init__(self, obs_dim: int, action_space: spaces.Box, hidden_dim: int = 64):
        super().__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.fc_mean = nn.Linear(hidden_dim, np.prod(action_space.shape))
        self.fc_logstd = nn.Linear(hidden_dim, np.prod(action_space.shape))
        self.critic = nn.Linear(hidden_dim, 1)  # Value head

        self.register_buffer(
            "action_scale",
            torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32),
        )

    def forward(self, x: any) -> tuple:
        """Forward pass through the network."""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        shared_features = self.shared_layer(x)
        mean = self.fc_mean(shared_features)
        log_std = self.fc_logstd(shared_features)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        value = self.critic(shared_features)
        return mean, log_std, value


# %%
def worker_fn(worker_id: int, 
              steps_per_worker: int,
              model_state_dict: dict,
              data_queue: mp.Queue,
              seed: int,
              scenic_file: str,
              central_learning : bool = True) -> None:
    """Execute function for each worker process. Initializes environment and model, collects trajectories, and sends data back."""
    
    logging.getLogger(__name__).debug("Worker %s: Initializing...", worker_id)
    
    agents = ['agent0', 'agent1']

    root_user = os.path.expanduser("~")
    sumo_map = root_user + "/ScenicGymClean/assets/maps/CARLA/Town04.net.xml"
    obs_space_dict = {"agent0" :  gym.spaces.Box(-0.0, 1.0 , (252,), dtype=np.float32),
                     "agent1": gym.spaces.Box(-0.0, 1.0 , (252,), dtype=np.float32)}
    # print(f"local decpared obs shape: {obs_space_dict['agent0'].shape}")
    action_space_dict = {'agent0': gym.spaces.Box(-1.0, 1.0, (2,), np.float32),
                         'agent1': gym.spaces.Box(-1.0, 1.0, (2,), np.float32)}
    
    scenario = scenic.scenarioFromFile(scenic_file,
                                   model="scenic.simulators.metadrive.model",
                               mode2D=True)

    env = ScenicZooEnv(scenario, 
                       MetaDriveSimulator(sumo_map=sumo_map, render=False, real_time=False),
                       None, 
                       max_steps=50, 
                       observation_space = obs_space_dict, 
                       action_space = action_space_dict, 
                       agents=agents)
    
    # TODO modify the code below to work with multi-agent things
    obs_space_shapes = dict()
    action_space_shapes = dict()
    obs_dims = dict()

    for agent in agents:
        obs_space_shape = env.observation_space(agent).shape
        # print(f"local obs space shape: {obs_space_shape}")
        # print(f"local env obs space shape: {env.observation_space(agent).shape}")
        action_space_shape = env.action_space(agent).shape

        obs_space_shapes[agent] = obs_space_shape 
        action_space_shapes[agent] = action_space_shape
        obs_dim = np.prod(obs_space_shape) if isinstance(obs_space_shape, tuple) else obs_space_shape[0]

        obs_dims[agent] = obs_dim

    
    worker_seed = seed + worker_id
    env.reset(seed=worker_seed)

    # the question here would be to have a central policy or independent
    # let's do central, so we at least converge
    # TODO doing central learning, should implement some decentralized learning in the future 
    default_agent = agents[0] 
    obs_dim = obs_dims[default_agent]
    action_space = action_space_dict[default_agent]
    
    # guess we are doing self-play
    # print(f"LOCAL obs_dim {obs_dim}")
    # print(f"LOCAL action_space {action_space}")
    local_model = ActorCritic(obs_dim, action_space)
    # print(f"LOCAL ACTORCRITIC: {local_model}")
    local_model.load_state_dict(model_state_dict)
    local_model.eval()

    observations = dict()
    actions = dict()
    log_probs = dict()
    rewards = dict()
    dones = []
    values = dict() 
    max_deviations = []

    for agent in agents:
        observations[agent] = [] 
        actions[agent] = [] 
        log_probs[agent] = [] 
        rewards[agent] = [] 
        # dones[agent] = [] 
        values[agent] = [] 

    obs, _ = env.reset()
    current_step = 0
    while current_step < steps_per_worker:
        # print(current_step)
        step_action_dict = dict()
        # log_prob_dict = dict()
        # value_dict = dict()

        for agent, o in obs.items():
            # print(f"LOCAL RAW OBS: {o}")
            # print(f"LOCAL RAW OBS FLAT: {o.flatten()}")
            obs_tensor = torch.tensor(o.flatten(), dtype=torch.float32).unsqueeze(0)
            # print(f"LOCAL OBS TENSOR: {obs_tensor}")

            with torch.no_grad():
                # print(f"LOCAL OBS TENSOR: {obs_tensor.shape}")
                mean, log_std, value = local_model(obs_tensor)
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                x_t = normal.rsample()
                y_t = torch.tanh(x_t)
                action = y_t * local_model.action_scale + local_model.action_bias
                log_prob = normal.log_prob(x_t)
                log_prob -= torch.log(local_model.action_scale * (1 - y_t.pow(2)) + 1e-6)
                log_prob = log_prob.sum(1, keepdim=True)

                # log_prob_dict[agent] = log_prob

                action = action.cpu().numpy().squeeze(0)
                step_action_dict[agent] = action
                # print(f"VALUE SHAPE {value.shape}")
                # print(f"VALUE ITEM {value.item()}")
                values[agent].append(value.item()) # FIXME need more permanent sol
                log_probs[agent].append(log_prob.item())
                actions[agent].append(action)
                observations[agent].append(o.flatten())
        
        next_obs, reward, terminated, truncated, _ = env.step(step_action_dict)
        # print(f"STEPPING")
        done = terminated or truncated
        dones.append(done)

        # TODO figure out if these go inside the loop over agents
        # figure out if the order of appending them matters
        for agent in agents:
            rewards[agent].append(reward[agent])
            # observations[agent].append(obs[agent].flatten())
            # actions[agent].append(step_action_dict[agent])
            # log_probs[agent].append(log_prob_dict[agent].item())
            # dones[agent].append(done[agent])
            # values[agent].append(value_dict[agent].item())

        obs = next_obs
        current_step += 1

        if done:
            # print(f'done reset')
            # print(f"MAX DEV: {env.max_deviation}")
            max_deviations.append(env.max_deviation)
            obs, _ = env.reset()
    

    # TODO the above have made dictionaries out of the requried traj data
    # still need to address the parts below
    # last_value_dict = dict()
    for agent, o in obs.items():
        last_obs_tensor = torch.tensor(o.flatten(), dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            _, _, last_value = local_model(last_obs_tensor)
            last_value = last_value.item()

        # last_value_dict[agent] = last_value
        # print("computing traj data")
        trajectory_data = {
            "observations": np.array(observations[agent], dtype=np.float32),
            "actions": np.array(actions[agent], dtype=np.float32),
            "log_probs": np.array(log_probs[agent], dtype=np.float32),
            "rewards": np.array(rewards[agent], dtype=np.float32),
            "dones": np.array(dones, dtype=np.bool_),
            "values": np.array(values[agent], dtype=np.float32),
            "last_value": last_value,
            "last_done": done,
            "max_deviations": max_deviations
        }
        # print(f"TRAJ DATA: {trajectory_data}")

        data_queue.put(trajectory_data)

    logging.getLogger(__name__).info("Worker %s: Finished collecting %s steps.", worker_id, current_step)
    # print("closing!")
    env.close()
    # print("closed")
    sentinel = "SENTINEL"
    data_queue.put(sentinel)
    return
    # print("closed")


# %%
def compute_gae(
    rewards: np.array,
    values: np.array,
    dones: np.array,
    last_value: float,
    last_done: float,
    gamma: float,
    gae_lambda: float,
) -> tuple:
    """Compute Generalized Advantage Estimation (GAE)."""
    # print("INSIDE GAE")
    advantages = np.zeros_like(rewards)
    # print(f"advantages: {advantages}")
    last_gae_lam = 0
    num_steps = len(rewards)
    next_values = np.append(values[1:], last_value if not last_done else 0.0)
    next_non_terminal = 1.0 - dones
    deltas = rewards + gamma * next_values * next_non_terminal - values
    # print("before for loop")
    # print(f"num_steps: {num_steps}")
    # breakpoint()
    for t in reversed(range(num_steps)):
        # print(f"t value: {t}")
        last_gae_lam = deltas[t] + gamma * gae_lambda * next_non_terminal[t] * last_gae_lam
        # print(f"last_gae_lam: {last_gae_lam}")
        # print(f"last_gae_lam type: {type(last_gae_lam)}")
        # print()
        advantages[t] = last_gae_lam

    returns = advantages + values
    return advantages, returns


# %%
def ppo_update(
    model: nn.Module,
    optimizer: optim.Optimizer,
    batch_obs: torch.Tensor,
    batch_actions: torch.Tensor,
    batch_log_probs_old: torch.Tensor,
    batch_advantages: torch.Tensor,
    batch_returns: torch.Tensor,
    num_epochs: int,
    minibatch_size: int,
    clip_epsilon: float,
    entropy_coef: float,
    value_loss_coef: float,
    max_grad_norm: float,
    rng: np.random.Generator,
) -> None:
    """Perform the PPO update step using collected batch data."""
    batch_size = batch_obs.size(0)
    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

    for _ in range(num_epochs):
        indices = rng.permutation(batch_size)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            minibatch_indices = indices[start:end]

            mb_obs = batch_obs[minibatch_indices]
            mb_actions = batch_actions[minibatch_indices]
            mb_log_probs_old = batch_log_probs_old[minibatch_indices]
            mb_advantages = batch_advantages[minibatch_indices]
            mb_returns = batch_returns[minibatch_indices]

            mean, log_std, values_pred = model(mb_obs)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)

            mb_actions_clamped = torch.clamp(mb_actions, -1.0 + EPSILON, 1.0 - EPSILON)
            unsquashed_mb_actions = torch.atanh(mb_actions_clamped)
            log_probs_gaussian = normal.log_prob(unsquashed_mb_actions).sum(dim=-1)
            log_prob_squash_correction = torch.log(1.0 - mb_actions.pow(2) + EPSILON).sum(dim=-1)
            log_probs_new = log_probs_gaussian - log_prob_squash_correction

            entropy = normal.entropy().mean()
            values_pred = values_pred.squeeze(-1)

            prob_ratio = torch.exp(log_probs_new - mb_log_probs_old)
            surr1 = prob_ratio * mb_advantages
            surr2 = torch.clamp(prob_ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * ((values_pred - mb_returns) ** 2).mean()

            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()


def main() -> None:
    """Run the PPO training."""
    args = tyro.cli(Args)

    all_max_devs = []

    # Ensure model directory exists
    if not pathlib.Path.exists(pathlib.Path(args.model_dir)):
        pathlib.Path.mkdir(pathlib.Path(args.model_dir))
    print("Model directory:", args.model_dir)

    # Set the environment name based on the scenic file
    env_name = pathlib.Path(args.scenic_file).stem

    # Set up multiprocess start method
    with contextlib.suppress(RuntimeError):
        mp.set_start_method("spawn")

    # seeds
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)

    logger.info("Environment: %s, Workers: %s, Total Timesteps: %s", env_name, args.num_workers, args.total_timesteps)

    obs_space_dict = {"agent0" :  gym.spaces.Box(-0.0, 1.0 , (252,), dtype=np.float32),
                     "agent1": gym.spaces.Box(-0.0, 1.0 , (252,), dtype=np.float32)}

    action_space_dict = {'agent0': gym.spaces.Box(-1.0, 1.0, (2,), np.float32),
                         'agent1': gym.spaces.Box(-1.0, 1.0, (2,), np.float32)}

    obs_space_shape = obs_space_dict["agent0"].shape
    action_space = action_space_dict["agent0"]

    obs_dim = np.prod(obs_space_shape) if isinstance(obs_space_shape, tuple) else obs_space_shape[0]
    # env.close()
    # print(f"OBS_DIM {obs_dim}")
    if args.use_pretrained:
        checkpoint = torch.load(args.checkpoint_model, weights_only=True)
        model = ActorCritic(obs_dim, action_space) 
        model.load_state_dict(checkpoint)
        model = model.to(device)
    else:
        model = ActorCritic(obs_dim, action_space).to(device)

    # model = ActorCritic(obs_dim, action_space).to(device)
    # print(f"GLOBAL MODEL {model}")

    batch_size = args.num_workers * args.steps_per_worker
    num_updates = args.total_timesteps // batch_size
    logger.info("Batch Size (Workers * Steps): %s", batch_size)
    logger.info("Total PPO Updates: %s", num_updates)

    data_queue = mp.Queue()

    total_steps = 0
    start_time = time.time()
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    total_episodes = 0
    total_drift = 0

    # Training Loop
    all_episode_rewards = []

    for update in range(1, num_updates + 1):
        update_start_time = time.time()
        model.eval()

        processes = []
        current_model_state_dict = model.state_dict()
        for i in range(args.num_workers):
            p = mp.Process(
                target=worker_fn,
                args=(
                    i,
                    args.steps_per_worker,
                    current_model_state_dict,
                    data_queue,
                    args.seed + update * args.num_workers,
                    args.scenic_file,
                ),
            )
            # print("STARTING process")
            p.start()
            # print('process started')
            processes.append(p)
        # print("gathering traj data")

        sentinel_num = 0
        all_trajectory_data = []

        while sentinel_num < args.num_workers:
            traj_data = data_queue.get()

            if traj_data == "SENTINEL":
                sentinel_num += 1
                continue

            all_trajectory_data.append(traj_data) 

        # print(f"processes num: {len(processes)}")

        for p in processes:
            p.join()
            # print("joined")
        logger.debug("Update %s: All workers finished.", update)


        for data in all_trajectory_data:
            current_episode_reward = 0
            current_episode_length = 0
            for reward, done in zip(data["rewards"], data["dones"]):
                current_episode_reward += reward
                current_episode_length += 1
                if done:
                    all_episode_rewards.append(current_episode_reward)
                    episode_rewards.append(current_episode_reward)
                    episode_lengths.append(current_episode_length)
                    total_episodes += 1
                    current_episode_reward = 0
                    current_episode_length = 0
        
        # num_episodes = 0
        # total_dev = 0
        for data in all_trajectory_data:
            max_dev = data["max_deviations"]
            all_max_devs.append(max_dev)
            # total_drift += np.sum(max_dev)
            # num_episodes += len(max_dev)

        # episode_mean_max_dev = total_dev/num_episodes

        # print(f"EPISODE MEAN MAX DEV: {episode_mean_max_dev}")


        total_steps += batch_size
        update_end_time = time.time()
        fps = int(batch_size / (update_end_time - update_start_time))
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0 # need this
        avg_length = np.mean(episode_lengths) if episode_lengths else 0

    end_time = time.time()
    final_eval_arr = np.array(all_episode_rewards)
    mean_episode_max_drift = np.mean(all_max_devs) 
    stddev_episode_max_drift = np.std(all_max_devs)
    episode_max_drift_stddev = np.std(all_max_devs)
    # np.save(args.eval_results_folder + "/" + args.eval_result_file_name, final_eval_arr)
    episode_mean_reward = np.mean(final_eval_arr)
    episode_reward_stddev = np.std(final_eval_arr)

    print(f"MEAN REWARD: {episode_mean_reward}") 
    print(f"MEAN EPISODE DRIFT {mean_episode_max_drift}")

    all_eval_results = {"episode_rewards" : final_eval_arr,
                        "episode_mean_reward" : episode_mean_reward,
                        "episode_reward_stddev" : episode_reward_stddev,
                        "mean_episode_drift" : mean_episode_max_drift,
                        "stddev_episode_max_drift" : stddev_episode_max_drift,
                        "all_episode_max_dev" : all_max_devs}

    pickle_filename = args.eval_results_folder + "/" + args.eval_result_file_name + ".pkl"

    with open(pickle_filename, 'wb') as file:
        pickle.dump(all_eval_results, file)

    logger.info(f"Eval finished in {end_time - start_time} seconds. Eval results pickled to {pickle_filename}")


if __name__ == "__main__":
    main()
