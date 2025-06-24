from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from pprint import pprint
import os
import numpy as np
import scenic
# from scenic.simulators.metadrive import MetaDriveSimulator
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback, CallbackList, CheckpointCallback
import datetime
import argparse
from env_utils import scenic_env, make_subproc_vec_env


parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model", type=str)
parser.add_argument("-e", "--eval", action="store_true")
parser.add_argument("-v", "--view", action="store_true")
parser.add_argument("-ee", "--eval_episodes", type=int)

args = parser.parse_args()


if __name__ == "__main__":
    assert args.model is not None, "You did not specify a model to eval"
    env = scenic_env()
    model = PPO.load(args.model, env=env)

    if args.eval:
        assert parser.eval_episodes is not None, "You did not specify how many episodes to run eval"
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=args.eval_episodes)

    if args.view:
        vec_env = model.get_env()
        obs = vec_env.reset()
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = vec_env.step(action)
            if any(dones):
                break


