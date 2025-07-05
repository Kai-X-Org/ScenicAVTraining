import numpy as np
import matplotlib.pyplot as plt
import dill
import argparse

def load_dill(file_name):

    with open(file_name, "rb") as f:
        res = dill.load(f)

    return res 

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str) # the model whose dill files you would like to graph
parser.add_argument("-s", "--save", action="store_true")
parser.add_argument("-r", "--resumed", action="store_true") # if a model should be graphed resumed from a pre-trained model
parser.add_argument("-pm", "--pretrained_model", type=str)

args = parser.parse_args()

assert args.model is not None, "You did not specify a model"
if args.r:


dill_dir = f"ray_models/{args.model}/dill"
checkpoint_name = lambda i: f"{dill_dir}/checkpoint_{i}.pkl"

num_steps = 0
agent_returns = dict(agent0=list(),
                     agent1=list()) 
env_timesteps = list()

for i in range(20):
    file_name = checkpoint_name(i)
    res = load_dill(file_name)

    num_steps += res['env_runners']['num_env_steps_sampled']  
    env_timesteps.append(num_steps)

    returns_dict = res['env_runners']['agent_episode_returns_mean']  

    agent_returns['agent0'].append(returns_dict['agent0'])
    agent_returns['agent1'].append(returns_dict['agent1'])
    
assert len(env_timesteps) == len(agent_returns['agent0']), "mismatch between timesteps list length and returns list length"
assert len(env_timesteps) == len(agent_returns['agent1']), "mismatch between timesteps list length and returns list length"

plt.figure()
plt.title(f"Training Curve of {args.model}")
plt.plot(env_timesteps, agent_returns['agent0'], label='agen0')
plt.xlabel("Num Steps")
plt.ylabel("Avg Episode Return")
plt.plot(env_timesteps, agent_returns['agent1'], label='agent1')
plt.legend()
plt.show()

if args.save:
    plt.savefig(f"{dill_dir}/{args.model}.png")

# FIXME those are dictionaries, need to enter the agent name

# agent_mean_returns = res['env_runners']['agent_episode_returns_mean']
# steps = res['env_runners']['num_env_steps_sampled'] # each is 4000 steps

