import dill
import os
import argparse

def load_dill(file_name):

    with open(file_name, "rb") as f:
        res = dill.load(f)
    return res



parser = argparse.ArgumentParser()

parser.add_argument("-s", "--scenic_program", type=str) # the Scenic progra that is the name of the folder

args = parser.parse_args()

assert args.scenic_program is not None

eval_dir = f"ray_models/eval_results/{args.scenic_program}"

filenames = os.listdir(eval_dir)
for f in filenames:
    res = load_dill(f"{eval_dir}/{f}")
    print(f"Eval Results for {f} Agent 0:\n{res['mean']['agent0']} +/- {res['std']['agent0']}")
    print(f"Eval Results for {f} Agent 1:\n{res['mean']['agent1']} +/- {res['std']['agent1']}")
    # print(f"Eval Results for {f}:\n{res['std']}")

