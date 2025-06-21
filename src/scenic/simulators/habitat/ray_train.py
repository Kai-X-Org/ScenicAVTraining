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

    return env


register_env("scenic", lambda cfg: scenic_env())

if __name__ == "__main__":
    # env = scenic_env()


    # register_env("scenic", lambda cfg: scenic_env())

    now = datetime.datetime.now()    
    now = now.strftime("%m_%d_%H_%M")

    config = (PPOConfig()
              # .get_default_config()
              .environment("scenic")
              # .training(lr=0.0002,
                # train_batch_size_per_learner=256,
                # num_epochs=10,)
              .rl_module(
                  model_config=DefaultModelConfig(
                      conv_filters = [
                          [16, 4, 2],
                          [32, 4, 2],
                          [64, 4, 2],
                          [128, 4, 2]
                      ],
                      conv_activation = 'relu',
                      head_fcnet_hiddens=[256],
                  )
              )
              .env_runners(num_env_runners=12,
                           gym_env_vectorize_mode=("ASYNC"))

            )

    config.evaluation(
        # Run one evaluation round every iteration.
        evaluation_interval=1,

        # Create 2 eval EnvRunners in the extra EnvRunnerGroup.
        evaluation_num_env_runners=2,

        # Run evaluation for exactly 10 episodes. Note that because you have
        # 2 EnvRunners, each one runs through 5 episodes.
        evaluation_duration_unit="episodes",
        evaluation_duration=10,
    )

    ppo = config.build_algo()
    current_dir = os.getcwd()
    for i in range(20):
        pprint(f"Checkpoint info: \n {ppo.train()} \n")
        checkpoint_dir = ppo.save_to_path(current_dir + f"/ray_models/test_checkpoints_{now}_{i}")
        pprint(f"CHECKPOINT SAVED TO PATH: {checkpoint_dir}\n\n")
