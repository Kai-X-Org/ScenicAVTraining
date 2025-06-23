from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
# from scenic.zoo import ScenicZooEnv
from scenic.gym import ScenicGymEnv
# from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pprint import pprint
import os
import numpy as np
import scenic
# from scenic.simulators.metadrive import MetaDriveSimulator
from scenic.simulators.habitat import HabitatSimulator
import datetime
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback, CallbackList, CheckpointCallback
import datetime
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model", type=str)
parser.add_argument("-e", "--eval", action="store_true")
parser.add_argument("-v", "--view", action="store_true")

args = parser.parse_args()

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

if __name__ == "__main__":
    assert args.model is not None, "You did not specify a model to eval"
    model = PPO.load(args.model, )
    if args.eval:


