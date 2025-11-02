import os
import time
import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
import imageio

# -----------------------------
# Settings
# -----------------------------
model_path = "highway_fast_dqn_normal/model"  # path to your trained model
output_dir = "gifs_normal"                      # folder for GIFs
num_episodes = 5                              
max_steps_per_episode = 1000                  
fps = 14                                      # playback speed

# Create folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Helper: get next available GIF number
# -----------------------------
def get_next_gif_number(folder):
    existing = [f for f in os.listdir(folder) if f.endswith(".gif")]
    numbers = [int(f.split("_")[1].split(".")[0]) for f in existing if "_" in f]
    return max(numbers) + 1 if numbers else 1

# -----------------------------
# Load environment and model
# -----------------------------
env = gym.make("highway-fast-v0", render_mode="rgb_array")

# ✅ unwrap before configuring
env = env.unwrapped
env.configure({
    "lanes_count": 3,         # increase number of lanes
    "vehicles_count": 20,
    "duration": 30,
    "offscreen_rendering": False,
     "reward_speed_range": [30, 40],  # higher target speeds
    "right_lane_reward": 0.1,         # small positive reward for right lane
    "collision_reward": -10,
    "lane_change_reward": 0.05,
})

env.reset()

model = DQN.load(model_path)

# -----------------------------
# Run episodes and save GIFs
# -----------------------------
for _ in range(num_episodes):
    obs, info = env.reset()
    done = truncated = False
    frames = []
    ep_reward = 0
    step_count = 0

    while not (done or truncated) and step_count < max_steps_per_episode:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        ep_reward += reward

        # Collect frame for GIF
        frame = env.render()
        frames.append(frame)

        # Live playback
        time.sleep(1 / fps)
        step_count += 1

    # Determine next available GIF filename
    gif_number = get_next_gif_number(output_dir)
    gif_path = os.path.join(output_dir, f"episode_{gif_number}.gif")
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"✅ Episode finished with reward {ep_reward:.2f}, saved to {gif_path}")

env.close()
