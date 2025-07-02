import numpy as np
import matplotlib.pyplot as plt

model_dir = "sb_models/"
eval_file = "evaluations.npz"

data_dirs = ["habitat_nav_06_23_19_35_eval", "habitat_nav_06_24_02_50_eval", "habitat_nav_06_24_11_51_eval"]

data_lst = [np.load(model_dir + d + "/" + eval_file) for d in data_dirs]

timesteps = np.array([])
rewards = np.array([])

step_count = 0
for d in data_lst:
    # print(d["timesteps"])
    # print(d["results"])
    # print(f"{step_count}")
    timesteps = np.hstack((timesteps, d["timesteps"] + step_count))
    rewards = np.hstack((rewards, np.mean(d["results"], axis=-1)))
    step_count += timesteps[-1]


assert rewards.shape == timesteps.shape
print(f"Timesteps: {timesteps}")
plt.figure()
plt.plot(timesteps, rewards)
plt.show()

