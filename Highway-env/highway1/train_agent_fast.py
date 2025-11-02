import os
import time
import gymnasium as gym
import highway_env
from stable_baselines3 import DQN
import imageio

# -----------------------------
# Create directories
# -----------------------------
model_dir = "highway_fast_dqn_normal"
os.makedirs(model_dir, exist_ok=True)

gif_dir = "normal_gifs_fast"
os.makedirs(gif_dir, exist_ok=True)

# -----------------------------
# Training environment
# -----------------------------
train_env = gym.make("highway-fast-v0")
train_env = train_env.unwrapped  # unwrap so we can access config

# Now customize reward weights
train_env.config["right_lane_reward"] = 1.0
train_env.config["lane_change_reward"] = 0.1
train_env.config["collision_reward"] = -10
train_env.reset()

# -----------------------------
# Train DQN model
# -----------------------------
model = DQN(
    "MlpPolicy",
    train_env,
    policy_kwargs=dict(net_arch=[256, 256]),
    learning_rate=5e-4,
    buffer_size=15000,
    learning_starts=200,
    batch_size=32,
    gamma=0.8,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=50,
    verbose=1,
    tensorboard_log="highway_fast_dqn_normal/",
    device="auto",  # use GPU if available
)

# Train for longer to improve performance
model.learn(int(5e4))  # increase timesteps if needed
model.save(os.path.join(model_dir, "model"))

train_env.close()

# -----------------------------
# Evaluation environment
# -----------------------------
eval_env = gym.make("highway-fast-v0", render_mode="rgb_array")
model = DQN.load(os.path.join(model_dir, "model"))

num_episodes = 5
for ep in range(1, num_episodes + 1):
    obs, info = eval_env.reset()
    done = truncated = False
    frames = []
    ep_reward = 0

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        ep_reward += reward

        # Collect frame for GIF
        frame = eval_env.render()
        frames.append(frame)

        # Render live ~14 FPS
        time.sleep(1/14)

    # Save GIF for this episode
    gif_path = os.path.join(gif_dir, f"episode_{ep}.gif")
    imageio.mimsave(gif_path, frames, fps=14)
    print(f"✅ Episode {ep} finished with reward {ep_reward:.2f}, saved to {gif_path}")

eval_env.close()
