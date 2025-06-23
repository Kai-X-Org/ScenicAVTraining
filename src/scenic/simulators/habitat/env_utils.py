import gymnasium as gym
from scenic.gym import ScenicGymEnv
import os
import numpy as np
import scenic
from scenic.simulators.habitat import HabitatSimulator
from stable_baselines3.common.vec_env import SubprocVecEnv

def scenic_env():

    root_user = os.path.expanduser("~")
    obs_space = gym.spaces.Box(0, 255, (256, 256, 3), np.uint8)
    action_space = gym.spaces.Box(-2.0, 2.0, (2,), np.float32) # habitat uses +/- 20.0, but let's just use 5

    scenic_file = "train_scenes/train.scenic"
    # print("Making Scenario")

    scenario = scenic.scenarioFromFile(scenic_file,
                                   model="scenic.simulators.habitat.model",
                               mode2D=False)

    # print("Made Scenario")
    env = ScenicGymEnv(scenario, 
                       HabitatSimulator(),
                       render_mode=None, 
                       max_steps=150, 
                       observation_space=obs_space, 
                       action_space=action_space
                       )
    env = gym.wrappers.RecordEpisodeStatistics(env)

    return env

def make_subproc_vec_env(num_envs):
    return SubprocVecEnv([scenic_env for _ in range(num_envs)])

