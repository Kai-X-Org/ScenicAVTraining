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
import datetime
import argparse
# import pickle
import dill

parser = argparse.ArgumentParser()

parser.add_argument("-f", "--file", type=str) # the scenic file used for training
parser.add_argument("-r", "--resume", action="store_true") # resuming training from an existing model?
parser.add_argument("-m", "--model", type=str) # from which model to resume training
parser.add_argument("-n", "--name", type=str) # what to name the model dir
parser.add_argument("-e", "--epochs", type=int) # first set of experiments are 20 epochs

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
    sumo_map = root_user + "/ScenicGymClean/assets/maps/CARLA/Town04.net.xml"
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
                       MetaDriveSimulator(sumo_map=sumo_map, render=False, real_time=False),
                       None, 
                       max_steps=50, 
                       observation_space = obs_space_dict, 
                       action_space = action_space_dict, 
                       agents=agents,
                       feedback_fn = max_cum_reward)
    # print("ENV CREATED!!!!!")
    return env


# register_env("scenic", lambda cfg: ParallelPettingZooEnv(scenic_env()))

if __name__ == "__main__":
    model_name = "train"
    args = parser.parse_args()
    assert args.epochs is not None, "You need to specify how many epochs you want to run training"

    assert args.file is not None, "You did not specify a Scenic program for training"
    if args.resume:
        assert args.model is not None, "You did not provide a model from which to resume training"

    register_env("scenic", lambda cfg: ParallelPettingZooEnv(scenic_env(args.file)))

    now = datetime.datetime.now()    
    now = now.strftime("%m_%d_%H_%M")

    if args.name is not None:
        model_name = args.name

    model_name = f"{model_name}_{now}"

    config = (PPOConfig()
              # .get_default_config()
              .environment("scenic")
              .multi_agent(
                  policies = {"p0"},
                  policy_mapping_fn = (lambda aid, *args, **kwargs: "p0"),
              )
              # .training(lr=0.0002,
                # train_batch_size_per_learner=256,
                # num_epochs=10,)
              # .rl_module()
              # checkpointing(export_native_model_files=True)
              .env_runners(num_env_runners=12)

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

    current_dir = os.getcwd()
    ppo = config.build_algo()
    if args.resume:
        ppo.restore_from_path(current_dir + "/" + args.model)
    # print("FINISHED ALGO BUILD!")
    # ppo.save_to_path("ray_models/")
    # current_dir = os.getcwd()
    # checkpoint_dir = ppo.save_to_path(current_dir + f"/ray_models/test_checkpoints_{now}")
    # pickle_dir = current_dir + f"/ray_models/{model_name}/pickles/"
    dill_dir = current_dir + f"/ray_models/{model_name}/dill/"
    os.makedirs(dill_dir, exist_ok=True)

    for i in range(args.epochs):
        progress_dict = ppo.train()
        # print(type(progress_dict))
        # pprint(f"Checkpoint info: \n {progress_dict} \n")
        checkpoint_dir = ppo.save_to_path(current_dir + f"/ray_models/{model_name}/checkpoint_{i}")
        dill_filename = dill_dir + f"checkpoint_{i}.pkl"

        with open(dill_filename, mode='wb') as dill_file:
            dill.dump(progress_dict, dill_file)

        # we should get an evaluation thing going on here!

        pprint(f"CHECKPOINT SAVED TO PATH: {checkpoint_dir}\n\n")

