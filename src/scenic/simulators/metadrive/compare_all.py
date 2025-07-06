import numpy as np
import matplotlib.pyplot as plt
import dill
import argparse
from graph_utils import get_train_curve

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--agent", type=str)

args = parser.parse_args()
assert args.agent is not None, "Which agent's performance do you want to graph?"

pretrained_model = "uniform_pretrain_07_03_22_06"

models = dict(bo="bo_finetune_40e_07_05_00_52", 
          halton="halton_finetune_40e_07_04_20_05", 
          mab="mab_finetune_40e_07_05_12_59",
          uniform="uniform_finetune_40e_07_05_14_20")

plt.figure()
plt.title(f"{args.agent} Train Curve")
plt.xlabel("Num Steps")
plt.ylabel("Avg Episode Return")

for m in models.keys():
    env_timesteps, agent_returns = get_train_curve(models[m], True, 33, pretrained_model=pretrained_model)
    plt.plot(env_timesteps, agent_returns[args.agent], label=m)

plt.legend()
plt.show()

