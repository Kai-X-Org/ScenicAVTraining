import dill
import os
import argparse

def load_dill(file_name):

    with open(file_name, "rb") as f:
        res = dill.load(f)
    return res



parser = argparse.ArgumentParser()

# parser.add_argument("-s", "--scenic_program", type=str) # the Scenic progra that is the name of the folder
parser.add_argument("-m", "--model_name", type=str)
parser.add_argument("-o", "--output_name", type=str)

args = parser.parse_args()

assert args.model_name is not None

eval_dir = f"ray_models/eval_results/uniform"

mean_rewards = []
std_rewards = []
for i in range(20):
    res = load_dill(f"{eval_dir}/{args.model_name}_{i}.pkl")
    mean_rewards.append(res["mean"])
    std_rewards.append(res["std"])

d = dict(mean=mean_rewards, std=std_rewards)

with open(f"{eval_dir}/{args.output_name}.pkl", "wb") as f:
    dill.dump(d, f)
    
# filenames = os.listdir(eval_dir)
# for f in filenames:
    # res = load_dill(f"{eval_dir}/{f}")
    # print(f"Eval Results for {f} Agent 0:\n{res['mean']['agent0']} +/- {res['std']['agent0']}")
    # print(f"Eval Results for {f} Agent 1:\n{res['mean']['agent1']} +/- {res['std']['agent1']}")
    # print(f"Eval Results for {f}:\n{res['std']}")

