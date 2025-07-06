import os
import numpy as np
import torch
from ray.rllib.core.rl_module import RLModule
import dill
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str) # which model?
parser.add_argument("-c", "--checkpoint", type=int) # which checkpoint?
parser.add_argument("-sf", "--scenic_file", type=str) # which scenic program?

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

def scenic_env(scenic_file):
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

if __name__ == '__main__':
    args = parser.parse_args()
    assert args.model is not None, "You did not specify a model to evaluate"
    assert args.checkpoint is not None, "You did not specify which checkpoint to use"
    assert args.scenic_file is not None, "You did not specify a scenic_file to evaluate"

    current_dir = os.getcwd()

    model_dir = f"{current_dir}/ray_models/{args.model}/checkpoint_{args.checkpoint}"

    rl_module = RLModule.from_checkpoint(
        Path(model_dir)
        / "learner_group"
        / "learner"
        / "rl_module"
        / "p0"
    )

    env = scenic_env(args.scenic_file)
    episode_return = 0.0
    done = False

    obs, info = env.reset()
    # FIXME need to run more than one episode
    while not done:

        # Compute the next action from a batch (B=1) of observations.
        obs0 = obs['agent0']
        obs1 = obs['agent1']
        obs_batch0 = torch.from_numpy(obs0).unsqueeze(0)  # add batch B=1 dimension
        obs_batch1 = torch.from_numpy(obs1).unsqueeze(0)  # add batch B=1 dimension
        # TODO check if this works for multi-agent
        model_outputs0 = rl_module.forward_inference({"obs": obs_batch0})
        model_outputs1 = rl_module.forward_inference({"obs": obs_batch1})

        # Extract the action distribution parameters from the output and dissolve batch dim.
        action_dist_params0 = model_outputs0["action_dist_inputs"][0].numpy()
        action_dist_params1 = model_outputs1["action_dist_inputs"][0].numpy()

        # We have continuous actions -> take the mean (max likelihood).
        greedy_action0 = np.clip(
            action_dist_params0[0:1],  # 0=mean, 1=log(stddev), [0:1]=use mean, but keep shape=(1,)
            a_min=env.action_space.low[0],
            a_max=env.action_space.high[0],
        )
        greedy_action1 = np.clip(
            action_dist_params1[0:1],  # 0=mean, 1=log(stddev), [0:1]=use mean, but keep shape=(1,)
            a_min=env.action_space.low[0],
            a_max=env.action_space.high[0],
        )
        breakpoint()
        # For discrete actions, you should take the argmax over the logits:
        # greedy_action = np.argmax(action_dist_params)

        # Send the action to the environment for the next step.
        obs, reward, terminated, truncated, info = env.step(greedy_action)

        # Perform env-loop bookkeeping.
        episode_return += reward
        done = terminated or truncated
        # TODO record stddev and mean across episodes run
