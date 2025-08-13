import numpy as np
import matplotlib.pyplot as plt
import dill
import argparse

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

        returns_dict = res['evaluation']['env_runners']['agent_episode_returns_mean']  

        agent_returns['agent0'].append(returns_dict['agent0'])
        agent_returns['agent1'].append(returns_dict['agent1'])

    return num_steps


def get_train_curve(model : str, resumed : bool, epochs : int, save : bool = False, pretrained_model : str = ""):
    if resumed:
        assert pretrained_model is not "", "You did not specify a starting model to graph"
        
    dill_dir = f"ray_models/{model}/dill"
    checkpoint_name = lambda i: f"{dill_dir}/checkpoint_{i}.pkl"

    agent_returns = dict(agent0=list(),
                         agent1=list()) 
    env_timesteps = list()

    # Can actually generalize this for a list of checkpoints...but let's do that for a later date
    pretrain_num_steps = 0

    if resumed:
        pretrain_num_steps = process_results(pretrained_model, env_timesteps, agent_returns, 20)

    num_steps = process_results(model, env_timesteps, agent_returns, epochs, num_steps_start=pretrain_num_steps)


    assert len(env_timesteps) == len(agent_returns['agent0']), "mismatch between timesteps list length and returns list length"
    assert len(env_timesteps) == len(agent_returns['agent1']), "mismatch between timesteps list length and returns list length"

    return env_timesteps, agent_returns

    # plt.figure()

    # if args.resumed:
        # plt.title(f"Training Curve of {model}, Resumed at Step {pretrain_num_steps}")
    # else:
        # plt.title(f"Training Curve of {model}")

    # # plt.title(f"Training Curve of {args.model}")

    # plt.plot(env_timesteps, agent_returns['agent0'], label='agen0')
    # plt.xlabel("Num Steps")
    # plt.ylabel("Avg Episode Return")
    # plt.plot(env_timesteps, agent_returns['agent1'], label='agent1')
    # plt.legend()
    # plt.show()

    # if args.save:
        # plt.savefig(f"{dill_dir}/{model}.png")

# def plot_for_agent(agent : str, env_timesteps, agent_returns):
    # plt.figure()

    # plt.plot(env_timesteps, agent_returns['agent0'], label='agen0')
    # plt.xlabel("Num Steps")
    # plt.ylabel("Avg Episode Return")
    # plt.plot(env_timesteps, agent_returns['agent1'], label='agent1')
    # plt.legend()
    # plt.show()

    # if args.save:
        # plt.savefig(f"{dill_dir}/{model}.png")

def get_seed_avg_train_curve(models, num_epochs, resumed):
    agent0 = 'agent0'
    agent1 = 'agent1'
    pretrained_model = 'uniform_pretrain_20_uniform_eval_08_11_16_38'

    # performance_dict = dict(env_timesteps=dict(), agent_returns=dict())
    performance_trace = dict(agent0=[], agent1=[])
    timesteps_trace = dict(agent0=[], agent1=[])

    # models = ["uniform_finetune_20e_uniform_eval_s0_08_12_13_27",
              # "uniform_finetune_20e_uniform_eval_s1_08_12_13_57",
              # "uniform_finetune_20e_uniform_eval_s2_08_12_14_29",
              # "uniform_finetune_20e_uniform_eval_s3_08_12_15_01",
              # "uniform_finetune_20e_uniform_eval_s4_08_12_15_31"]

    for m in models:
        dill_dir = f"ray_models/{m}/dill"
        checkpoint_name = lambda i: f"{dill_dir}/checkpoint_{i}.pkl"

        agent_returns = dict(agent0=list(),
                             agent1=list()) 
        env_timesteps = list()

        # Can actually generalize this for a list of checkpoints...but let's do that for a later date
        pretrain_num_steps = 0

        if resumed:
            pretrain_num_steps = process_results(pretrained_model,
                                                 env_timesteps, 
                                                 agent_returns,
                                                 20)

        num_steps = process_results(m,
                                    env_timesteps,
                                    agent_returns,
                                    num_epochs,
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
    
    return env_timesteps, dict(agent0=agent0_avg_returns,
                               agent1=agent1_avg_returns)

    # breakpoint()
    # plt.figure()

    # plt.title(f"Training Curve")

    # plt.plot(env_timesteps, agent0_avg_returns, label='agent0')
    # plt.xlabel("Num Steps")
    # plt.ylabel("Avg Episode Return")
    # plt.plot(env_timesteps, agent1_avg_returns, label='agent1')
    # plt.legend()
    # plt.show()

