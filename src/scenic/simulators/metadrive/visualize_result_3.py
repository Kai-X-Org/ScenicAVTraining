import dill
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def load_dill(file_name):

    with open(file_name, "rb") as f:
        res = dill.load(f)
    return res


res_dir = "ray_models/eval_results/uniform"

samplers = ["bo", "mab", "halton", "uniform"]
seeds = [i for i in range(5)]

res_dict = dict()
agent0 = 'agent0'
agent1 = 'agent1'

times = [i * 4000 for i in range(41)]


res = load_dill(f"{res_dir}/pretrain.pkl")['mean']
a1 = [0.0]
a2 = [0.0]
d = dict(agent0=[], agent1=[])

for r in res:
    a1.append(r[agent0])
    a2.append(r[agent1])

d[agent0].append(a1)
d[agent1].append(a2)

res_dict['pretrain'] = d
# breakpoint()
# print(f"{d}")


for s in samplers:
    d = dict(agent0=[], agent1=[])
    for i in seeds:
        res = load_dill(f"{res_dir}/{s}_s{i}.pkl")['mean']
        # print(f"RES: {res}")
        a1 = []
        a2 = []
        for r in res:
            a1.append(r[agent0])
            a2.append(r[agent1])
        d[agent0].append(a1)
        d[agent1].append(a2)

    res_dict[s] = d

# print(f"{res_dict}")
    # print(f"{d[agent0]}")
# print(f"res dict {res_dict}")
# print(len(res_dict["halton"][agent0]))
for a in [agent0, agent1]:
    plt.figure()
    plt.title(f"Training of {s} Sampler Training")
    plt.xlabel("timesteps")
    plt.ylabel("rewards")
    for s in samplers:
        rewards = np.hstack((res_dict['pretrain'][a][0], np.mean(res_dict[s][a], axis=0)))
        plt.plot(times, rewards, label=s)
    plt.legend()
    plt.savefig(f"{a}_training.png") 
    

