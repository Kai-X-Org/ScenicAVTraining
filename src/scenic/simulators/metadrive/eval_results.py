import pickle
import argparse
import numpy as np
parser = argparse.ArgumentParser(description="Process Args")
parser.add_argument("--filename", "-f", help="sum the integers (default: find the max)")

args = parser.parse_args()

if __name__ == "__main__":
    filename = args.filename
    assert filename is not None, "You did not provide a file name"
    with open(filename, "rb") as file:
        loaded_dict = pickle.load(file)

    episode_reward = loaded_dict["episode_rewards"]
    mean_reward = np.mean(episode_reward)
    reward_stddev = np.std(episode_reward)

    all_episode_max_dev = loaded_dict["all_episode_max_dev"]
    mean_drift = np.mean(all_episode_max_dev)
    stddev_drift = np.std(all_episode_max_dev)

    print(f"Episode Reward Mean: {mean_reward}")
    print(f"Episode Reward StdDev: {reward_stddev}")
    print(f"Episode Max Drift Mean: {mean_drift}")
    print(f"Episode Max Drift StdDev: {stddev_drift}")

    

