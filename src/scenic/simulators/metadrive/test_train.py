import gymnasium as gym
from scenic.zoo import ScenicZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from pprint import pprint
import os
import numpy as np
import scenic
from scenic.simulators.metadrive import MetaDriveSimulator
import datetime
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("-f", "--file", type=str)
parser.add_argument("-r", "--resume", action="store_true")
parser.add_argument("-m", "--model", type=str)

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


# register_env("scenic", lambda cfg: ParallelPettingZooEnv(scenic_env()))

if __name__ == "__main__":

    args = parser.parse_args()

    assert args.file is not None, "You did not specify a Scenic program for training"
    if args.resume:
        assert args.model is not None, "You did not provide a model from which to resume training"

    env = scenic_env(args.file)
    action = [1, 0]
    action_dict = dict(agent0=action, agent1=action)
    for _ in range(4):
        obs, info = env.reset()
        for _ in range(1000):
            obs, reward, term, trunc, info = env.step(action_dict)
            if any(term.values()) or any(trunc.values()):
                # breakpoint()
                print(f"term: {term}, trunc: {trunc}")
                break




