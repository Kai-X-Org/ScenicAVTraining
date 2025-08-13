import numpy as np
import matplotlib.pyplot as plt
import dill
import argparse
from graph_utils import get_seed_avg_train_curve

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--agent", type=str)

args = parser.parse_args()
assert args.agent is not None, "Which agent's performance do you want to graph?"


models = dict(bo=["bo_finetune_20e_uniform_eval_s0_08_11_18_07",
                  "bo_finetune_20e_uniform_eval_s1_08_11_18_46",
                  "bo_finetune_20e_uniform_eval_s2_08_11_19_33",
                  "bo_finetune_20e_uniform_eval_s3_08_11_20_24",
                  "bo_finetune_20e_uniform_eval_s4_08_11_21_06"], 
          halton=["halton_finetune_20e_uniform_eval_s0_08_11_21_46",
                  "halton_finetune_20e_uniform_eval_s1_08_11_22_29",
                  "halton_finetune_20e_uniform_eval_s2_08_11_23_11",
                  "halton_finetune_20e_uniform_eval_s3_08_11_23_43",
                  "halton_finetune_20e_uniform_eval_s4_08_12_00_49"], 
          mab=["mab_finetune_20e_uniform_eval_s0_08_12_01_19",
               "mab_finetune_20e_uniform_eval_s1_08_12_01_48",
               "mab_finetune_20e_uniform_eval_s2_08_12_02_18",
               "mab_finetune_20e_uniform_eval_s3_08_12_02_48",
               "mab_finetune_20e_uniform_eval_s4_08_12_03_18"],
          uniform= ["uniform_finetune_20e_uniform_eval_s0_08_12_13_27",
              "uniform_finetune_20e_uniform_eval_s1_08_12_13_57",
              "uniform_finetune_20e_uniform_eval_s2_08_12_14_29",
              "uniform_finetune_20e_uniform_eval_s3_08_12_15_01",
              "uniform_finetune_20e_uniform_eval_s4_08_12_15_31"]
              )

plt.figure()
plt.title(f"{args.agent} Train Curve")
plt.xlabel("Num Steps")
plt.ylabel("Avg Episode Return")

for m in models.keys():
    env_timesteps, agent_returns = get_seed_avg_train_curve(models[m], 20, True)
    plt.plot(env_timesteps, agent_returns[args.agent], label=m)

plt.legend()
plt.show()

