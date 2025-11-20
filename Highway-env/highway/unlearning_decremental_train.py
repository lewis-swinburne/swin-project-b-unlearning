#!/usr/bin/env python3
"""
unlearn_decremental.py
- Performs short decremental unlearning on DQN.
- Logs CPU, GPU, mean reward, unlearning rate, latency, and elapsed time.
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
# Config
# -----------------------------
original_model_path = "highway_fast_dqn_normal/model.zip"
unlearn_dir = "highway_fast_dqn_unlearn_decremental"
os.makedirs(unlearn_dir, exist_ok=True)

TOTAL_TIMESTEPS = 50_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5
UNLEARN_FRACTION = 0.5

# -----------------------------
# Environment
# -----------------------------
def make_env():
    env = gym.make("highway-fast-v0")
    env = env.unwrapped
    env.configure({
        "lanes_count": 4,
        "vehicles_count": 20,
        "duration": 30,
        "offscreen_rendering": False,
        "reward_speed_range": [20, 30],
    })
    return env

# -----------------------------
# Logging Callback
# -----------------------------
class EvalAndLogCallbackDec(BaseCallback):
    def __init__(self, eval_env, eval_freq, n_eval_episodes, csv_path):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.csv_path = csv_path
        self.start_time = None

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
                self.model, self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
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
            print(f"[DecEval] Step={self.num_timesteps} MeanR={mean_r:.2f} CPU={cpu}% GPU={gpu}% Rate={rate:.1f} steps/s")
        return True

# -----------------------------
# Main
# -----------------------------
def main():
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log_dir = os.path.join(unlearn_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    eval_csv_path = os.path.join(log_dir, f"decremental_eval_{timestamp}.csv")

    train_env = make_env()
    train_env = Monitor(train_env)
    eval_env = make_env()

    print(f"Loading model from: {original_model_path}")
    model = DQN.load(original_model_path, env=train_env, device="cuda" if torch.cuda.is_available() else "cpu")

    # Decremental unlearning (replay buffer cut)
    if hasattr(model, "replay_buffer"):
        current_size = model.replay_buffer.size()
        n_remove = int(current_size * UNLEARN_FRACTION)
        if n_remove > 0:
            print(f"Removing {n_remove} transitions from replay buffer...")
            model.replay_buffer.pos = max(0, model.replay_buffer.pos - n_remove)
            model.replay_buffer.size = max(0, model.replay_buffer.size - n_remove)

    callback = EvalAndLogCallbackDec(eval_env, EVAL_FREQ, N_EVAL_EPISODES, eval_csv_path)

    print(f"🚀 Starting decremental unlearning for {TOTAL_TIMESTEPS:,} timesteps...")
    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    elapsed = time.time() - t0

    model_path = os.path.join(unlearn_dir, f"model_decremental_{timestamp}")
    model.save(model_path)
    print(f"✅ Decremental unlearning done in {round(elapsed,1)}s. Model saved to {model_path}.zip")
    print(f"CSV metrics: {eval_csv_path}")

if __name__ == "__main__":
    main()
