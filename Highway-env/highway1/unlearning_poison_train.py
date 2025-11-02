#!/usr/bin/env python3
"""
unlearn_poisoning.py
- Performs short reward-poisoning unlearning on a DQN model.
- Logs CPU, GPU, latency, mean rewards, timesteps, elapsed time, and unlearning rate.
"""

import os
import time
import csv
from datetime import datetime
import gymnasium as gym
import numpy as np
import psutil
import torch
import highway_env
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

# -----------------------------
# Reward Poisoning Wrapper
# -----------------------------
class RewardPoisonWrapper(gym.Wrapper):
    def __init__(self, env, poison_prob=0.3, flip_sign=True, scale=1.0):
        super().__init__(env)
        self.poison_prob = poison_prob
        self.flip_sign = flip_sign
        self.scale = scale

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if np.random.rand() < self.poison_prob:
            reward = -reward * self.scale if self.flip_sign else reward * self.scale
        return obs, reward, done, truncated, info

# -----------------------------
# Config
# -----------------------------
original_model_path = "highway_fast_dqn_normal/model.zip"
unlearn_dir = "highway_fast_dqn_unlearn_poisoning"
os.makedirs(unlearn_dir, exist_ok=True)

TOTAL_TIMESTEPS = 50_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5
POISON_PROB = 0.3
POISON_FLIP = True
POISON_SCALE = 1.0

# -----------------------------
# Environment factory
# -----------------------------
def make_env(poison=False):
    env = gym.make("highway-fast-v0")
    env = env.unwrapped
    env.configure({
        "lanes_count": 4,
        "vehicles_count": 20,
        "duration": 30,
        "offscreen_rendering": False,
        "reward_speed_range": [20, 30],
    })
    if poison:
        env = RewardPoisonWrapper(env, poison_prob=POISON_PROB, flip_sign=POISON_FLIP, scale=POISON_SCALE)
    return env

# -----------------------------
# Logging Callback
# -----------------------------
class EvalAndLogCallbackPoison(BaseCallback):
    def __init__(self, eval_env, eval_freq, n_eval_episodes, csv_path):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.csv_path = csv_path
        self.start_time = None

        # Init CSV
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestep", "mean_reward", "std_reward", "cpu_usage(%)",
                "gpu_usage(%)", "unlearning_rate(steps/s)", "elapsed_seconds", "latency(s)"
            ])

    def _get_gpu_usage(self):
        if torch.cuda.is_available():
            try:
                return torch.cuda.utilization(torch.cuda.current_device())
            except Exception:
                return 0
        return 0

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            t0 = time.time()
            rewards, _ = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes,
                warn=False, return_episode_rewards=True
            )
            latency = (time.time() - t0) / self.n_eval_episodes
            mean_r, std_r = np.mean(rewards), np.std(rewards)
            cpu = psutil.cpu_percent(interval=None)
            gpu = self._get_gpu_usage()
            elapsed = time.time() - self.start_time
            rate = self.num_timesteps / elapsed if elapsed > 0 else 0

            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    self.num_timesteps, mean_r, std_r, cpu, gpu,
                    round(rate, 2), round(elapsed, 2), round(latency, 3)
                ])
            print(f"[PoisonEval] Step={self.num_timesteps} MeanR={mean_r:.2f} CPU={cpu}% GPU={gpu}% Rate={rate:.1f} steps/s")
        return True

# -----------------------------
# Main
# -----------------------------
def main():
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log_dir = os.path.join(unlearn_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    eval_csv_path = os.path.join(log_dir, f"poison_eval_{timestamp}.csv")

    print(f"Loading model from: {original_model_path}")
    train_env = make_env(poison=True)
    eval_env = make_env(poison=False)
    model = DQN.load(original_model_path, env=train_env, device="cuda" if torch.cuda.is_available() else "cpu")

    callback = EvalAndLogCallbackPoison(eval_env, EVAL_FREQ, N_EVAL_EPISODES, eval_csv_path)
    print(f"🚀 Starting poisoning unlearning for {TOTAL_TIMESTEPS:,} timesteps...")
    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    elapsed = time.time() - t0

    # Save
    model_path = os.path.join(unlearn_dir, f"model_poisoned_{timestamp}")
    model.save(model_path)
    print(f"✅ Poison unlearning done in {round(elapsed,1)}s. Model saved to {model_path}.zip")
    print(f"CSV metrics: {eval_csv_path}")

if __name__ == "__main__":
    main()
