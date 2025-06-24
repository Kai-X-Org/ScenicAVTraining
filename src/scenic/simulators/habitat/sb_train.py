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

parser.add_argument('-m', '--model', type=str)
parser.add_argument('-r', '--resume', action="store_true")
parser.add_argument('-t', '--timesteps', type=int)
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
    
    assert args.timesteps is not None, "You did not provide how many timesteps to train"

    if args.resume:
        assert args.model is not None, "You did not provide a model from which to resume training"

    if args.model is not None:
        assert args.resume, "You provided a model path, but did not set the resume training flag, -r"

    print(f"Resume training: {args.resume}, model dir: {args.model}, timesteps: {args.timesteps}")

    now = datetime.datetime.now()    
    now = now.strftime("%m_%d_%H_%M")
    model_folder_name = f"habitat_nav_{now}" 

    env = SubprocVecEnv([scenic_env for _ in range(5)])
    # eval_env = SubprocVecEnv([scenic_env for _ in range(6)])
    eval_env = scenic_env()
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"./sb_models/{model_folder_name}_eval",
                             log_path=f"./sb_models/{model_folder_name}_eval", eval_freq=10_000,
                             deterministic=True, render=False)
    if args.resume:
        print(f"Loading model from {args.model}")
        model = PPO.load(args.model, env=env)
    else:
        model = PPO("CnnPolicy", env, verbose=2)

    model.learn(total_timesteps=args.timesteps, callback=eval_callback, progress_bar=True)
    model.save(f"./sb_models/trained_models/{model_folder_name}")

    print(f"Time Stamp: {now}")

    if args.resume:
        print(f"Resumed Training From {args.model}")

