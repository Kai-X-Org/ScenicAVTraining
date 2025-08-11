import os
import numpy as np
import torch
from ray.rllib.core.rl_module import RLModule
import dill
from pathlib import Path
import argparse
import gymnasium as gym
import scenic
from scenic.zoo import ScenicZooEnv
import dill

from scenic.simulators.metadrive import MetaDriveSimulator
# from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
# from ray.tune.registry import register_env
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str) # which model?
parser.add_argument("-c", "--checkpoint", type=int) # which checkpoint?
parser.add_argument("-sf", "--scenic_file", type=str) # which scenic program?

def max_cum_reward(result):
    agent0_return = result.records["agent0_return"]
    agent1_return = result.records["agent1_return"]

    return max(agent0_return, agent1_return)

def cum_reward(result):
    """
    For the mab sampler
    """
    agent0_return = result.records["agent0_return"]
    agent1_return = result.records["agent1_return"]

    return [agent0_return, agent1_return] 

def scenic_env(scenic_file):
    agents = ['agent0', 'agent1']

    root_user = os.path.expanduser("~")
    sumo_map = root_user + "/ScenicGymClean/assets/maps/CARLA/Town04.net.xml"
    obs_space_dict = {"agent0" :  gym.spaces.Box(-0.0, 1.0 , (252,), dtype=np.float32),
                     "agent1": gym.spaces.Box(-0.0, 1.0 , (252,), dtype=np.float32)}
    # print(f"local decpared obs shape: {obs_space_dict['agent0'].shape}")
    action_space_dict = {'agent0': gym.spaces.Box(-1.0, 1.0, (2,), np.float32),
                         'agent1': gym.spaces.Box(-1.0, 1.0, (2,), np.float32)}

    # scenic_file = "exp_uniform.scenic"
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
                       agents=agents,
                       feedback_fn = max_cum_reward)
    # print("ENV CREATED!!!!!")
    return env

def run_eval(env, model_dir, model_name, checkpoint_num):
    agent0 = 'agent0'
    agent1 = 'agent1'

    current_dir = os.getcwd()

    # model_dir = f"{current_dir}/ray_models/{args.model}/checkpoint_{args.checkpoint}"

    rl_module = RLModule.from_checkpoint(
        Path(model_dir)
        / "learner_group"
        / "learner"
        / "rl_module"
        / "p0"
    )

    # env = scenic_env(scenic_file)
    all_returns_dict = dict(agent0=list(), agent1=list())
    # TODO SET SEEDS!!!
    for seed in range(3):
        for i in range(30):
            if i == 0:
                obs, info = env.reset(seed=seed)
            else:
                obs, info = env.reset()

            episode_return = dict(agent0=0.0, agent1=0.0)
            done = False
            while not done:

                # Compute the next action from a batch (B=1) of observations.
                obs0 = obs[agent0]
                obs1 = obs[agent1]
                obs_batch0 = torch.from_numpy(obs0).unsqueeze(0)  # add batch B=1 dimension
                obs_batch1 = torch.from_numpy(obs1).unsqueeze(0)  # add batch B=1 dimension
                # TODO check if this works for multi-agent
                model_outputs0 = rl_module.forward_inference({"obs": obs_batch0})
                model_outputs1 = rl_module.forward_inference({"obs": obs_batch1})

                # Extract the action distribution parameters from the output and dissolve batch dim.
                action_dist_params0 = model_outputs0["action_dist_inputs"][0].numpy()
                action_dist_params1 = model_outputs1["action_dist_inputs"][0].numpy()

                # We have continuous actions -> take the mean (max likelihood).
                greedy_action0 = [np.clip(
                    action_dist_params0[i],  # 0=mean, 1=log(stddev), [0:1]=use mean, but keep shape=(1,)
                    a_min=env.action_space(agent0).low[0],
                    a_max=env.action_space(agent0).high[0],
                ) for i in [0, 1]]
                greedy_action1 = [np.clip(
                    action_dist_params1[i],  # 0=mean, 1=log(stddev), [0:1]=use mean, but keep shape=(1,)
                    a_min=env.action_space(agent1).low[0],
                    a_max=env.action_space(agent1).high[0],
                ) for i in [0, 1]]

                # For discrete actions, you should take the argmax over the logits:
                # greedy_action = np.argmax(action_dist_params)

                # Send the action to the environment for the next step.
                action_dict = dict(agent0 = greedy_action0, agent1 = greedy_action1)
                obs, reward, terminated, truncated, info = env.step(action_dict)

                # Perform env-loop bookkeeping.
                episode_return[agent0] += reward[agent0]
                episode_return[agent1] += reward[agent1]
                done = any(terminated.values()) or any(truncated.values())

            all_returns_dict[agent0].append(episode_return[agent0])
            all_returns_dict[agent1].append(episode_return[agent1])

    mean_return = dict()

    mean_return[agent0] = np.mean(all_returns_dict[agent0])
    mean_return[agent1] = np.mean(all_returns_dict[agent1])

    std_return = dict()
    std_return[agent0] = np.std(all_returns_dict[agent0])
    std_return[agent1] = np.std(all_returns_dict[agent1])

    print(f"MEAN RETURNS: {mean_return}")
    print(f"Std RETURNS: {std_return}")

    result_dict = dict(mean=mean_return, std=std_return)
    
    dill_dir = f"{current_dir}/ray_models/eval_results/uniform"
    os.makedirs(dill_dir, exist_ok = True)

    dill_file = f"{dill_dir}/{model_name}_{checkpoint_num}.pkl"
    with open(dill_file, mode='wb') as f:
        dill.dump(result_dict, f)

