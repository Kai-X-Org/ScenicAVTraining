import gymnasium as gym
from scenic.zoo import ScenicZooEnv
from pprint import pprint
import os
import numpy as np
import scenic
from scenic.simulators.carla import CarlaSimulator
import datetime
import argparse
import dill

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

def scenic_env(scenic_file, stored_scene_data=None):
    agents = ['agent0', 'agent1']

    root_user = os.path.expanduser("~")
    sumo_map = root_user + "/ScenicGymClean/assets/maps/CARLA/Town04.xodr"
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
                       CarlaSimulator("Town04", 
                                      sumo_map,
                                      render=False),
                       None, 
                       max_steps=50, 
                       observation_space = obs_space_dict, 
                       action_space = action_space_dict, 
                       agents=agents,
                       feedback_fn = max_cum_reward)
    # print("ENV CREATED!!!!!")
    return env
