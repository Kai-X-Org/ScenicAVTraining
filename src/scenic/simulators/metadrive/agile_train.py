from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from scenic.zoo import ScenicZooEnv
from scenic.simulators.metadrive import MetaDriveSimulator
import os
import scenic
import gymnasium as gym
import numpy as np

num_envs = 2

def make_env():
    def thunk():
        agents = ['agent0', 'agent1']

        root_user = os.path.expanduser("~")
        sumo_map = root_user + "/ScenicGym/assets/maps/CARLA/Town04.net.xml"
        obs_space_dict = {"agent0" :  gym.spaces.Box(-0.0, 1.0 , (252,), dtype=np.float32),
                         "agent1": gym.spaces.Box(-0.0, 1.0 , (252,), dtype=np.float32)}
        # print(f"local decpared obs shape: {obs_space_dict['agent0'].shape}")
        action_space_dict = {'agent0': gym.spaces.Box(-1.0, 1.0, (2,), np.float32),
                             'agent1': gym.spaces.Box(-1.0, 1.0, (2,), np.float32)}

        scenic_file = "exp_local.scenic"
        print("Making Scenario")
       
        scenario = scenic.scenarioFromFile(scenic_file,
                                       model="scenic.simulators.metadrive.model",
                                   mode2D=True)
        
        print("Made Scenario")
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
    vec_env = AsyncPettingZooVecEnv(
          [
              make_env()
              for _ in range(num_envs - 1)
          ]
      )

    action = dict(agent0 = [0.1, 0.5], agent1=[0.1, 0.5])

    observations, infos = vec_env.reset()
    for step in range(25):
        actions = {
            agent: [action[agent] for _ in range(num_envs - 1)]
            for agent in vec_env.agents
        }
        observations, rewards, terminations, truncations, infos = vec_env.step(actions)
        print(f"rew {rewards}")
