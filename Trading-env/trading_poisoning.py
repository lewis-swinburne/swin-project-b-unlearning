import os
import re
import sys
import time
import gymnasium as gym
import gym_trading_env
import pandas as pd
import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque
import pickle
import json
import psutil
from datetime import datetime
import itertools
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = ''
device = torch.device("cpu")
print(f"[Device] Using: {device}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description="Poisoning-based unlearning experiment runner")
parser.add_argument("--epochs", nargs="+", type=int, required=True, help="List of unlearning epoch counts, e.g. --epochs 10 25 50")
parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="List of random seeds, e.g. --seeds 0 42 100")
parser.add_argument("--corruption", type=float, default=0.1, help="Corruption level for poison_state (default: 0.1)")
args = parser.parse_args()

EPOCH_LIST = args.epochs
SEED_LIST = args.seeds
CORRUPTION_LEVEL = args.corruption

print(f"[Config] Epochs to test: {EPOCH_LIST}")
print(f"[Config] Seeds to test: {SEED_LIST}")
print(f"[Config] Corruption level: {CORRUPTION_LEVEL}")

save_dir = "checkpoints"
unlearn_dir = "unlearning_results"
os.makedirs(unlearn_dir, exist_ok=True)

df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")
total_length = len(df)
forget_start = int(0.4 * total_length)
forget_end = int(0.6 * total_length)
print(f"[Data Split] Total: {total_length}, Forget set: {forget_start}-{forget_end}")

class UnlearningMonitor:
    def __init__(self, unlearn_epochs, seed):
        self.unlearn_epochs = unlearn_epochs
        self.seed = seed
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.epoch_logs = []
        self.overall_stats = {
            "start_timestamp": None,
            "end_timestamp": None,
            "total_duration_seconds": 0,
            "peak_cpu_percent": 0,
            "peak_memory_mb": 0,
            "average_cpu_percent": 0,
            "average_memory_mb": 0,
        }
        
    def start(self):
        self.start_time = datetime.now()
        self.overall_stats["start_timestamp"] = self.start_time.isoformat()
        print(f"[Monitor] Started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def log_epoch(self, epoch_num, epoch_duration_seconds):
        cpu_percent = self.process.cpu_percent(interval=0.1)
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        elapsed = (datetime.now() - self.start_time).total_seconds()
        epoch_data = {
            "epoch": epoch_num,
            "timestamp": datetime.now().isoformat(),
            "elapsed_time_seconds": round(elapsed, 2),
            "epoch_duration_seconds": round(epoch_duration_seconds, 2),
            "cpu_percent": round(cpu_percent, 2),
            "memory_mb": round(memory_mb, 2),
        }
        self.epoch_logs.append(epoch_data)
        if cpu_percent > self.overall_stats["peak_cpu_percent"]:
            self.overall_stats["peak_cpu_percent"] = round(cpu_percent, 2)
        if memory_mb > self.overall_stats["peak_memory_mb"]:
            self.overall_stats["peak_memory_mb"] = round(memory_mb, 2)
        return epoch_data

    def finish(self):
        end_time = datetime.now()
        self.overall_stats["end_timestamp"] = end_time.isoformat()
        self.overall_stats["total_duration_seconds"] = round(
            (end_time - self.start_time).total_seconds(), 2
        )
        if self.epoch_logs:
            avg_cpu = sum(log["cpu_percent"] for log in self.epoch_logs) / len(self.epoch_logs)
            avg_mem = sum(log["memory_mb"] for log in self.epoch_logs) / len(self.epoch_logs)
            self.overall_stats["average_cpu_percent"] = round(avg_cpu, 2)
            self.overall_stats["average_memory_mb"] = round(avg_mem, 2)
        print(f"\n[Monitor] Finished. Peak CPU: {self.overall_stats['peak_cpu_percent']}%, "
              f"Peak Mem: {self.overall_stats['peak_memory_mb']:.2f} MB")

    def save(self, output_dir):
        filepath = os.path.join(
            output_dir, f"monitoring_epochs{self.unlearn_epochs}_seed{self.seed}.json"
        )
        with open(filepath, "w") as f:
            json.dump({
                "unlearn_epochs": self.unlearn_epochs,
                "seed": self.seed,
                "overall_stats": self.overall_stats,
                "epoch_logs": self.epoch_logs,
            }, f, indent=4)
        print(f"[Monitor] Saved to: {filepath}")

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        try:
            s_copy = np.copy(state)
        except Exception:
            s_copy = state
        try:
            ns_copy = np.copy(next_state)
        except Exception:
            ns_copy = next_state
        self.buffer.append((s_copy, action, reward, ns_copy, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)
    def clear(self): self.buffer.clear()

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, output_dim)
    def forward(self, x):
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        return self.lin3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, buffer):
        self.model = DQN(state_dim, action_dim).to(device)
        self.target_model = DQN(state_dim, action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.buffer = buffer
        self.gamma = 0.99
        self.loss_fn = nn.MSELoss()
        self.action_dim = action_dim
        self.update_target_steps = 1000
        self.train_steps = 0

    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()
    
    def update(self, batch_size):
        if len(self.buffer) < batch_size:
            return
        state, action, reward, next_state, done = self.buffer.sample(batch_size)
        state = torch.FloatTensor(np.array(state)).to(device)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)
        action = torch.LongTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(done).to(device)
        q_values = self.model(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.model(next_state).argmax(1)
            next_q_values = self.target_model(next_state)
            next_q_value = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        loss = self.loss_fn(q_value, expected_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.train_steps += 1
        if self.train_steps % self.update_target_steps == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        loss = self.loss_fn(q_value, expected_q_value)
    
        q_mean = q_value.mean().item()
        loss_value = loss.item()
    
        return loss_value, q_mean  

    def save(self, filepath, epsilon):
        torch.save({
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
            "epsilon": epsilon,
        }, filepath)

    def load_original_model_only(self, filepath):
        checkpoint = torch.load(filepath, map_location=device)
        self.model.load_state_dict(checkpoint["model"])
        self.target_model.load_state_dict(checkpoint["target_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_steps = checkpoint.get("train_steps", 0)
        self.buffer.clear()
        return checkpoint.get("epsilon", 1.0)

def get_latest_checkpoint(directory="checkpoints", prefix="dqn_checkpoint_ep"):
    checkpoints = []
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith(".pth"):
            match = re.search(r"ep(\d+).pth", filename)
            if match:
                episode = int(match.group(1))
                checkpoints.append((episode, filename))
    if checkpoints:
        checkpoints.sort(reverse=True, key=lambda x: x[0])
        return os.path.join(directory, checkpoints[0][1])
    return None

def poison_state(state, corruption_level=0.1):
    state_copy = state.copy()
    n_features = len(state_copy)
    n_corrupt = int(n_features * corruption_level)
    if n_corrupt > 0:
        corrupt_indices = np.random.choice(n_features, n_corrupt, replace=False)
        corrupted_values = state_copy[corrupt_indices].copy()
        np.random.shuffle(corrupted_values)
        state_copy[corrupt_indices] = corrupted_values
    return state_copy

def evaluate_agent(agent, env, df_segment, num_episodes=30, epsilon=0.05): # 30 episodes, previously 10 but averages had high variance
    rewards = []
    for _ in range(num_episodes):
        state, info = env.reset()
        done, truncated = False, False
        total_reward = 0
        while not done and not truncated:
            action = agent.get_action(state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

for unlearn_epochs, seed in itertools.product(EPOCH_LIST, SEED_LIST):
    print("\n" + "="*90)
    print(f"[Run] Starting unlearning run: {unlearn_epochs} epochs | seed {seed}")
    print("="*90)

    set_seed(seed)
    print(f"[Config] Using random seed: {seed}")

    checkpoint_path = get_latest_checkpoint(directory=save_dir)
    if not checkpoint_path:
        print("[Error] No trained model found. Please train a model first.")
        sys.exit(1)
    print(f"[Load] Found checkpoint: {checkpoint_path}")

    env = gym.make("TradingEnv", name="BTCUSD", df=df, positions=[-1, 0, 1], trading_fees=0.01/100, borrow_interest_rate=0.0003/100)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    unlearn_buffer = ReplayBuffer(10000)
    agent = DQNAgent(state_dim, action_dim, unlearn_buffer)
    epsilon = agent.load_original_model_only(checkpoint_path)

    print("\n" + "-"*80)
    print(f"[Phase] Evaluating ORIGINAL model (before unlearning) | Epochs={unlearn_epochs} | Seed={seed}")
    print("-"*80)

    env_forget = gym.make("TradingEnv", name="BTCUSD", df=df.iloc[forget_start:forget_end], positions=[-1,0,1], trading_fees=0.01/100, borrow_interest_rate=0.0003/100)
    env_retain = gym.make("TradingEnv", name="BTCUSD", df=df.iloc[:forget_start], positions=[-1,0,1], trading_fees=0.01/100, borrow_interest_rate=0.0003/100)

    forget_reward_before, forget_std_before = evaluate_agent(agent, env_forget, df.iloc[forget_start:forget_end])
    retain_reward_before, retain_std_before = evaluate_agent(agent, env_retain, df.iloc[:forget_start])

    print(f"[Original Evaluation] Forget mean: {forget_reward_before:.2f} | Retain mean: {retain_reward_before:.2f}")

    print("\n" + "-"*80)
    print(f"[Phase] Starting UNLEARNING (poisoning-based) | Epochs={unlearn_epochs} | Seed={seed}")
    print("-"*80)

    monitor = UnlearningMonitor(unlearn_epochs, seed)
    monitor.start()

    run_start = time.time()        
    unlearn_start = time.time()    

    batch_size = 32
    epsilon_unlearn = 0.05 # changed from 0.5, since intial value was different to learning value
    corruption_level = CORRUPTION_LEVEL

    forget_buffer = ReplayBuffer(5000)
    retain_buffer = ReplayBuffer(5000)

    for epoch in range(unlearn_epochs):
        epoch_start = time.time()

        state, info = env_forget.reset()
        done, truncated = False, False
        forget_steps = 0
        while not done and not truncated:
            poisoned_state = poison_state(state, corruption_level)
            action = agent.get_action(poisoned_state, epsilon_unlearn)
            next_state, reward, done, truncated, info = env_forget.step(action)
            reward = np.clip(reward, -1, 1)
            poisoned_next_state = poison_state(next_state, corruption_level)
            forget_buffer.push(poisoned_state, action, reward, poisoned_next_state, done or truncated)
            if len(forget_buffer) >= batch_size:
                original_buffer = agent.buffer
                agent.buffer = forget_buffer
                agent.update(batch_size)
                agent.buffer = original_buffer
            state = next_state
            forget_steps += 1

        state, info = env_retain.reset()
        done, truncated = False, False
        retain_steps = 0
        while not done and not truncated and retain_steps < 500:
            action = agent.get_action(state, epsilon_unlearn * 0.5)
            next_state, reward, done, truncated, info = env_retain.step(action)
            reward = np.clip(reward, -1, 1)
            retain_buffer.push(state, action, reward, next_state, done or truncated)
            if len(retain_buffer) >= batch_size:
                original_buffer = agent.buffer
                agent.buffer = retain_buffer
                agent.update(batch_size)
                agent.buffer = original_buffer
            state = next_state
            retain_steps += 1

        epoch_time = time.time() - epoch_start
        epoch_data = monitor.log_epoch(epoch + 1, epoch_time)
        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1}] Time {epoch_time:.2f}s | Forget buf {len(forget_buffer)} | Retain buf {len(retain_buffer)}")

    unlearn_time_minutes = (time.time() - unlearn_start) / 60.0
    monitor.finish()

    print("\n" + "-"*80)
    print(f"[Phase] Evaluating UNLEARNED model (after unlearning) | Epochs={unlearn_epochs} | Seed={seed}")
    print("-"*80)

    forget_reward_after, forget_std_after = evaluate_agent(agent, env_forget, df.iloc[forget_start:forget_end])
    retain_reward_after, retain_std_after = evaluate_agent(agent, env_retain, df.iloc[:forget_start])

    forget_degradation = ((forget_reward_before - forget_reward_after) / abs(forget_reward_before)) * 100 if forget_reward_before != 0 else 0
    retain_preservation = (retain_reward_after / retain_reward_before) * 100 if retain_reward_before != 0 else 0

    print(f"[Metrics] Forget deg: {forget_degradation:.2f}% | Retain pres: {retain_preservation:.2f}%")

    unlearned_model_path = os.path.join(unlearn_dir, f"unlearned_model_epochs{unlearn_epochs}_seed{seed}.pth")
    agent.save(unlearned_model_path, epsilon_unlearn)

    benchmark_path = os.path.join(unlearn_dir, f"benchmark_epochs{unlearn_epochs}_seed{seed}.json")
    benchmark_results = {
        "unlearn_epochs": unlearn_epochs,
        "seed": seed,
        "unlearn_time_minutes": unlearn_time_minutes,
        "corruption_level": corruption_level,
        "metrics": {
            "forget_degradation_percent": forget_degradation,
            "retain_preservation_percent": retain_preservation,
        },
        "original_performance": {
            "forget_set_mean": forget_reward_before,
            "retain_set_mean": retain_reward_before,
        },
        "unlearned_performance": {
            "forget_set_mean": forget_reward_after,
            "retain_set_mean": retain_reward_after,
        }
    }
    with open(benchmark_path, "w") as f:
        json.dump(benchmark_results, f, indent=4)
    print(f"[Save] Results saved to {benchmark_path}")

    monitor.save(unlearn_dir)
    env_forget.close()
    env_retain.close()
    env.close()
    run_end = time.time()
    total_minutes = (run_end - run_start) / 60.0

    print(f"\n[Timing] Unlearning loop time: {unlearn_time_minutes:.2f} min | Total wall time: {total_minutes:.2f} min")
    print(f"[Run Complete] {unlearn_epochs} epochs | Seed {seed} finished successfully.")

print("\n[All Runs Complete]")
