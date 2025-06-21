import gymnasium as gym
# from scenic.zoo import ScenicZooEnv
from scenic.gym import ScenicGymEnv
from ray.rllib.algorithms.ppo import PPOConfig
# from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from pprint import pprint
import os
import numpy as np
import scenic
# from scenic.simulators.metadrive import MetaDriveSimulator
from scenic.simulators.habitat import HabitatSimulator
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
import datetime

def scenic_env():

    root_user = os.path.expanduser("~")
    obs_space = gym.spaces.Box(0, 255, (256, 256, 3), np.uint8)
    action_space = gym.spaces.Box(-5.0, 5.0, (2,), np.float32) # habitat uses +/- 20.0, but let's just use 5

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
    return env
action = [1,0]
env = scenic_env()
env.reset()

for i in range(3):
    print("New episode")
    for j in range(100):
        print(f"step number: {j}")
        o, r, d, t, info = env.step(action)
        if d or t:
            break
    env.reset()
    print("finished reset")
