import pickle
import matplotlib.pyplot as plt
import glob
import re
import os
import numpy as np

CHECKPOINT_DIR = "checkpoints"

checkpoint_files = sorted(
    glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_rewards_ep*.pkl")),
    key=lambda x: int(re.search(r"ep(\d+)", x).group(1)) if re.search(r"ep(\d+)", x) else 0
)

if not checkpoint_files:
    raise FileNotFoundError(f"No checkpoint reward files found in '{CHECKPOINT_DIR}'.")

episodes = []
avg_rewards = []

for fpath in checkpoint_files:
    match = re.search(r"ep(\d+)", fpath)
    episode = int(match.group(1)) if match else None

    with open(fpath, "rb") as f:
        rewards = pickle.load(f)
        avg_reward = np.mean(rewards)

    episodes.append(episode)
    avg_rewards.append(avg_reward)

plt.figure(figsize=(12, 6))
plt.plot(episodes, avg_rewards, marker="o", linestyle="-", label="Avg Reward per Checkpoint", alpha=0.85)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("DQN Trading Agent - Average Reward per Checkpoint")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("exports/checkpoint_avg_rewards_plot.png")

print(f"Plot saved as 'exports/checkpoint_avg_rewards_plot.png' using {len(checkpoint_files)} checkpoint files.")
