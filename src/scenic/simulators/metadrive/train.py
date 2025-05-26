from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from scenic.zoo import ScenicZooEnv
from scenic.simulators.metadrive import MetaDriveSimulator
import os
import scenic
import gymnasium as gym
import torch
import numpy as np
from tqdm import trange

from agilerl.algorithms import IPPO
from agilerl.utils.algo_utils import obs_channels_to_first

num_envs = 3

def make_env():
    def thunk():
        agents = ['agent0', 'agent1']

        root_user = os.path.expanduser("~")
        sumo_map = root_user + "/ScenicGymClean/assets/maps/CARLA/Town04.net.xml"
        obs_space_dict = {"agent0" :  gym.spaces.Box(-0.0, 1.0 , (252,), dtype=np.float32),
                         "agent1": gym.spaces.Box(-0.0, 1.0 , (252,), dtype=np.float32)}
        # print(f"local decpared obs shape: {obs_space_dict['agent0'].shape}")
        action_space_dict = {'agent0': gym.spaces.Box(-1.0, 1.0, (2,), np.float32),
                             'agent1': gym.spaces.Box(-1.0, 1.0, (2,), np.float32)}

        scenic_file = "exp_uniform.scenic"
        # print("Making Scenario")
       
        scenario = scenic.scenarioFromFile(scenic_file,
                                       model="scenic.simulators.metadrive.model",
                                   mode2D=True)
        
        # print("Made Scenario")
        env = ScenicZooEnv(scenario, 
                           MetaDriveSimulator(sumo_map=sumo_map, render=False, real_time=False),
                           None, 
                           max_steps=50, 
                           observation_space = obs_space_dict, 
                           action_space = action_space_dict, 
                           agents=agents)

        return env
    return thunk


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = AsyncPettingZooVecEnv(
          [
              make_env()
              for _ in range(num_envs)
          ]
      )

    # action = dict(agent0 = [0.1, 0.5], agent1=[0.1, 0.5])

    observations, infos = env.reset()

    # Configure the multi-agent algo input arguments
    observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
    action_spaces = [env.single_action_space(agent) for agent in env.agents]
    agent_ids = [agent_id for agent_id in env.agents]

    channels_last = False  # Flag to swap image channels dimension from last to first [H, W, C] -> [C, H, W]

    agent = IPPO(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        device=device,
    )

    # Define training loop parameters
    max_steps = 150_000  # Max steps

    while agent.steps[-1] < max_steps:
        state, info  = env.reset() # Reset environment at start of episode
        scores = np.zeros((num_envs, len(agent.shared_agent_ids)))
        completed_episode_scores = []
        steps = 0

        if channels_last:
            state = {
                agent_id: obs_channels_to_first(s)
                for agent_id, s in state.items()
            }

        for _ in range(agent.learn_step):

            states = {agent_id: [] for agent_id in agent.agent_ids}
            actions = {agent_id: [] for agent_id in agent.agent_ids}
            log_probs = {agent_id: [] for agent_id in agent.agent_ids}
            rewards = {agent_id: [] for agent_id in agent.agent_ids}
            dones = {agent_id: [] for agent_id in agent.agent_ids}
            values = {agent_id: [] for agent_id in agent.agent_ids}

            done = {agent_id: np.zeros(num_envs) for agent_id in agent.agent_ids}

            for idx_step in range(-(agent.learn_step // -num_envs)):

                # Get next action from agent
                action, log_prob, _, value = agent.get_action(obs=state, infos=info)

                # Clip to action space
                clipped_action = {}
                for agent_id, agent_action in action.items():
                    shared_id = agent.get_homo_id(agent_id)
                    actor_idx = agent.shared_agent_ids.index(shared_id)
                    agent_space = agent.action_space[agent_id]
                    if isinstance(agent_space, gym.spaces.Box):
                        if agent.actors[actor_idx].squash_output:
                            clipped_agent_action = agent.actors[actor_idx].scale_action(agent_action)
                        else:
                            clipped_agent_action = np.clip(agent_action, agent_space.low, agent_space.high)
                    else:
                        clipped_agent_action = agent_action

                    clipped_action[agent_id] = clipped_agent_action

                # Act in environment
                next_state, reward, termination, truncation, info = env.step(clipped_action)
                scores += np.array(list(reward.values())).transpose()
                # print(f"FIRS termination: {termination}")

                steps += num_envs

                next_done = {}
                # print(f"TRAIN DONES: {dones}")
                # print(f"TRAIN REWARDS {rewards}")
                for agent_id in agent.agent_ids:
                    states[agent_id].append(state[agent_id])
                    actions[agent_id].append(action[agent_id])
                    log_probs[agent_id].append(log_prob[agent_id])
                    rewards[agent_id].append(reward[agent_id])
                    dones[agent_id].append(done[agent_id])
                    values[agent_id].append(value[agent_id])
                    next_done[agent_id] = np.logical_or(termination[agent_id], truncation[agent_id]).astype(np.int8)

                if channels_last:
                    next_state = {
                        agent_id: obs_channels_to_first(s)
                        for agent_id, s in next_state.items()
                    }

                # Find which agents are "done" - i.e. terminated or truncated
                # print(f"TERMINATION: {termination}")
                temp_dones = {
                    agent_id: termination[agent_id] | truncation[agent_id]
                    for agent_id in agent.agent_ids
                }

                # Calculate scores for completed episodes
                for idx, agent_dones in enumerate(zip(*temp_dones.values())):
                    # print(f"AGENT_DONES {agent_dones}")
                    if all(agent_dones):
                        completed_score = list(scores[idx])
                        completed_episode_scores.append(completed_score)
                        agent.scores.append(completed_score)
                        scores[idx].fill(0)

                state = next_state
                done = next_done

            experiences = (
                states,
                actions,
                log_probs,
                rewards,
                dones,
                values,
                next_state,
                next_done,
            )

            # Learn according to agent's RL algorithm
            loss = agent.learn(experiences)

        agent.steps[-1] += steps
    agent.save_chekpoint("models/ippo_uniform_05_23_23_23")
