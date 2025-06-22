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
    env = gym.vector.AsyncVectorEnv([lambda: scenic_env() for _ in range(12)])
    env.reset()
    action = [[1,0] for _ in range(12)]
    
    for i in range(3):
        print("New episode")
        for j in range(100):
            print(f"step number: {j}")
            breakpoint()
            o, r, d, t, info = env.step(action)
            if all(d) or all(t):
                break
        env.reset()
        print("finished reset")
        
        

