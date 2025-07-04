import numpy as np
import matplotlib.pyplot as plt
import dill

def load_dill(file_name):

    with open(file_path, "rb") as f:
        dict1 = dill.load(f)

    return dict1

file_name = ""

dict1 = load_dill(file_name)
agent_mean_returns = dict1['env_runners']['agent_episode_returns_mean']
steps = dict1['env_runners']['num_env_steps_sampled'] # each is 4000 steps
