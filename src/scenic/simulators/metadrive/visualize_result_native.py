import numpy as np
import matplotlib.pyplot as plt
import dill
import argparse


"""
graphs the eval done using the training scenic program
"""

agent0 = 'agent0'
agent1 = 'agent1'

def load_dill(file_name):

    with open(file_name, "rb") as f:
        res = dill.load(f)

    return res 


def process_results(model_name, env_timesteps, agent_returns, num_epochs, num_steps_start=0):
    dill_dir = f"ray_models/{model_name}/dill"
    checkpoint_name = lambda i: f"{dill_dir}/checkpoint_{i}.pkl"

    num_steps = num_steps_start

    for i in range(num_epochs):
        file_name = checkpoint_name(i)
        res = load_dill(file_name)

        num_steps += res['env_runners']['num_env_steps_sampled']  
        env_timesteps.append(num_steps)

        returns_dict = res['env_runners']['agent_episode_returns_mean']  

        agent_returns['agent0'].append(returns_dict['agent0'])
        agent_returns['agent1'].append(returns_dict['agent1'])

    return num_steps



parser = argparse.ArgumentParser()
parser.add_argument("-m", "--models", nargs='+') # the model whose dill files you would like to graph
parser.add_argument("-s", "--save", action="store_true")
parser.add_argument("-r", "--resumed", action="store_true") # if a model should be graphed resumed from a pre-trained model
parser.add_argument("-pm", "--pretrained_model", type=str)
parser.add_argument("-e", "--epochs", type=int) # how many epochs we trained this policy (thus how many checkpoints)

args = parser.parse_args()

pretrained_model = 'uniform_pretrain_20_uniform_eval_08_11_16_38'
# assert args.model is not None, "You did not specify a model"
assert args.epochs is not None, "You need to specify how many epochs (num checkpoints) you trained the agent"
# if args.resumed:
    # assert args.pretrained_model is not None, "You did not specify a starting model to graph"

# performance_dict = dict(env_timesteps=dict(), agent_returns=dict())
performance_trace = dict(agent0=[], agent1=[])
timesteps_trace = dict(agent0=[], agent1=[])

models = ["uniform_finetune_20e_uniform_eval_s0_08_12_13_27",
          "uniform_finetune_20e_uniform_eval_s1_08_12_13_57",
          "uniform_finetune_20e_uniform_eval_s2_08_12_14_29",
          "uniform_finetune_20e_uniform_eval_s3_08_12_15_01",
          "uniform_finetune_20e_uniform_eval_s4_08_12_15_31"]

for m in models:
    dill_dir = f"ray_models/{m}/dill"
    checkpoint_name = lambda i: f"{dill_dir}/checkpoint_{i}.pkl"

    agent_returns = dict(agent0=list(),
                         agent1=list()) 
    env_timesteps = list()

    # Can actually generalize this for a list of checkpoints...but let's do that for a later date
    pretrain_num_steps = 0

    if args.resumed:
        pretrain_num_steps = process_results(pretrained_model,
                                             env_timesteps, 
                                             agent_returns,
                                             20)

    num_steps = process_results(m,
                                env_timesteps,
                                agent_returns,
                                args.epochs,
                                num_steps_start=pretrain_num_steps)

    assert len(env_timesteps) == len(agent_returns['agent0']), "mismatch between timesteps list length and returns list length"
    assert len(env_timesteps) == len(agent_returns['agent1']), "mismatch between timesteps list length and returns list length"

    performance_trace[agent0].append(agent_returns[agent0])
    performance_trace[agent1].append(agent_returns[agent1])
    timesteps_trace[agent0].append(env_timesteps)
    timesteps_trace[agent1].append(env_timesteps)
    


agent0_avg_returns = np.mean(performance_trace[agent0], axis=0)
agent1_avg_returns = np.mean(performance_trace[agent1], axis=0)
env_timesteps = timesteps_trace[agent0][0] # legit any one would work

# breakpoint()
plt.figure()

plt.title(f"Training Curve")

plt.plot(env_timesteps, agent0_avg_returns, label='agent0')
plt.xlabel("Num Steps")
plt.ylabel("Avg Episode Return")
plt.plot(env_timesteps, agent1_avg_returns, label='agent1')
plt.legend()
plt.show()

# if args.save:
    # # plt.savefig(f"{dill_dir}/{args.model}.png")
    # plt.savefig(f"fig/{args.model}_native_eval.png")
