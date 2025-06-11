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
# def make_env() -> callable:
    # def thunk() -> gym.Env:
        # ... # create the envioronments
        # # might not even need this...

        # agents = ['agent0', 'agent1']

        # root_user = os.path.expanduser("~")
        # sumo_map = root_user + "/ScenicGymClean/assets/maps/CARLA/Town04.net.xml"
        # obs_space_dict = {"agent0" :  gym.spaces.Box(-0.0, 1.0 , (252,), dtype=np.float32),
                         # "agent1": gym.spaces.Box(-0.0, 1.0 , (252,), dtype=np.float32)}
        # # print(f"local decpared obs shape: {obs_space_dict['agent0'].shape}")
        # action_space_dict = {'agent0': gym.spaces.Box(-1.0, 1.0, (2,), np.float32),
                             # 'agent1': gym.spaces.Box(-1.0, 1.0, (2,), np.float32)}

        # scenic_file = "exp_uniform.scenic"
        # # print("Making Scenario")
       
        # scenario = scenic.scenarioFromFile(scenic_file,
                                       # model="scenic.simulators.metadrive.model",
                                   # mode2D=True)
        
        # # print("Made Scenario")
        # env = ScenicZooEnv(scenario, 
                           # MetaDriveSimulator(sumo_map=sumo_map, render=False, real_time=False),
                           # None, 
                           # max_steps=50, 
                           # observation_space = obs_space_dict, 
                           # action_space = action_space_dict, 
                           # agents=agents)

        # env = ScenicGymEnv(...)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        # return env
def scenic_env():
    agents = ['agent0', 'agent1']

    root_user = os.path.expanduser("~")
    sumo_map = root_user + "/ScenicGymClean/assets/maps/CARLA/Town04.net.xml"
    obs_space_dict = {"agent0" :  gym.spaces.Box(-0.0, 1.0 , (252,), dtype=np.float32),
                     "agent1": gym.spaces.Box(-0.0, 1.0 , (252,), dtype=np.float32)}
    # print(f"local decpared obs shape: {obs_space_dict['agent0'].shape}")
    action_space_dict = {'agent0': gym.spaces.Box(-1.0, 1.0, (2,), np.float32),
                         'agent1': gym.spaces.Box(-1.0, 1.0, (2,), np.float32)}

    scenic_file = "exp_uniform.scenic"
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
                       agents=agents)
    # print("ENV CREATED!!!!!")
    return env


register_env("scenic", lambda cfg: ParallelPettingZooEnv(scenic_env()))

if __name__ == "__main__":

    # register_env("scenic", lambda cfg: scenic_env())

    config = (PPOConfig()
              # .get_default_config()
              .environment("scenic")
              .multi_agent(
                  policies = {"p0"},
                  policy_mapping_fn = (lambda aid, *args, **kwargs: "p0"),
              )
              .training(lr=0.0002,
                train_batch_size_per_learner=256,
                num_epochs=10,)
              # .rl_module()
              .env_runners(num_env_runners=12)

            )

    ppo = config.build_algo()
    ppo.save_to_path("ray_models/ray_train_1.pth")
    pprint(ppo.train())
    

    # for _ in range(4):
        # pass
        # # pprint(ppo.train())
