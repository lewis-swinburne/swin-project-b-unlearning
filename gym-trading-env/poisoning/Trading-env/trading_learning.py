import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import re
import gymnasium as gym
import gym_trading_env
import pandas as pd
import torch
import torch.nn as nn
device = torch.device("cpu")
print(f"[Device] Using: {device}")
import random
import numpy as np
from collections import deque
import pickle

save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True) 

df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")

env = gym.make("TradingEnv",
    name="BTCUSD",
    df=df,
    positions=[-1, 0, 1],
    trading_fees=0.01/100,
    borrow_interest_rate=0.0003/100,
)

def get_latest_checkpoint(directory=".", prefix="dqn_checkpoint_ep"):
    checkpoints = []
    
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith(".pth"):
            match = re.search(r"ep(\d+).pth", filename)
            if match:
                episode = int(match.group(1))
                checkpoints.append((episode, filename))
    
    if checkpoints:
        checkpoints.sort(reverse=True, key=lambda x: x[0])
        return checkpoints[0][1] 
    return None

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        try:
            s_copy = np.copy(state)
        except Exception:
            s_copy = state
        try:
            ns_copy = np.copy(next_state)
        except Exception:
            ns_copy = next_state

        self.buffer.append((
            s_copy,
            action,
            reward,
            ns_copy,
            done
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def to_list(self):
        return list(self.buffer)

    def from_list(self, lst):
        self.buffer = deque(lst, maxlen=self.capacity)

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

    def save(self, filepath, epsilon):
        main_checkpoint = {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
            "epsilon": epsilon,
        }
        torch.save(main_checkpoint, filepath)

        buffer_path = filepath.replace(".pth", ".buffer.pkl")
        with open(buffer_path, "wb") as f:
            pickle.dump(self.buffer.to_list(), f)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)

        self.model.load_state_dict(checkpoint["model"])
        self.target_model.load_state_dict(checkpoint["target_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_steps = checkpoint.get("train_steps", 0)

        buffer_path = filepath.replace(".pth", ".buffer.pkl")
        if os.path.exists(buffer_path):
            with open(buffer_path, "rb") as f:
                buf_list = pickle.load(f)
            try:
                cap = self.buffer.capacity
            except Exception:
                cap = max(len(buf_list), 10000)
            new_buffer = ReplayBuffer(cap)
            new_buffer.from_list(buf_list)
            self.buffer = new_buffer

        return checkpoint.get("epsilon", 1.0)


def main():
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    buffer = ReplayBuffer(10000)
    agent = DQNAgent(state_dim, action_dim, buffer)

    continue_episodes = 1000  # how many *new* episodes to train for each run
    batch_size = 32
    epsilon = 1.0
    eval_interval = 50
    eval_episodes = 5
    training_rewards_path = os.path.join(save_dir, "training_rewards.pkl")

    checkpoint_path = get_latest_checkpoint(directory=save_dir)
    start_episode = 0
    all_rewards = []

    if checkpoint_path:
        checkpoint_full_path = os.path.join(save_dir, checkpoint_path)
        match = re.search(r"ep(\d+)", checkpoint_path)
        last_episode = int(match.group(1)) if match else 0

        print(f"[Resume] Loading saved model from {checkpoint_full_path} (episode {last_episode})...")
        epsilon = agent.load(checkpoint_full_path)
        print(f"[Resume] Epsilon restored: {epsilon}")

        start_episode = last_episode
        print(f"[Resume] Continuing training from episode {start_episode + 1}")

        if os.path.exists(training_rewards_path):
            with open(training_rewards_path, "rb") as f:
                all_rewards = pickle.load(f)
            print(f"[Resume] Loaded {len(all_rewards)} recorded rewards.")
        else:
            print("[Resume] No previous reward file found — starting fresh reward list.")
    else:
        print("[Start] No checkpoint found. Starting new training.")
        all_rewards = []

    max_total_episodes = 4000  # stop training after this many total episodes
    # agent currently trained at 2000 - 27/10/25
    # agent currently trained at 3650 - 30/10/25

    if start_episode >= max_total_episodes:
        print(f"[Info] Reached max episode limit ({max_total_episodes}). Training complete.")
        env.close()
        exit()

    end_episode = min(start_episode + continue_episodes, max_total_episodes)
    for episode in range(start_episode + 1, end_episode + 1):
        state, info = env.reset()
        done, truncated = False, False
        total_reward = 0

        while not done and not truncated:
            action = agent.get_action(state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            reward = np.clip(reward, -1, 1)
            buffer.push(state, action, reward, next_state, done or truncated)
            agent.update(batch_size)
            state = next_state
            total_reward += reward

        epsilon = max(0.01, epsilon * 0.995)

        all_rewards.append(total_reward)
        print(f"[Episode {episode}] Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}")

        if episode % 50 == 0 or episode == start_episode + continue_episodes:
            checkpoint_name = os.path.join(save_dir, f"dqn_checkpoint_ep{episode}.pth")
            agent.save(checkpoint_name, epsilon)

            with open(training_rewards_path, "wb") as f:
                pickle.dump(all_rewards, f)

            checkpoint_rewards_path = os.path.join(save_dir, f"checkpoint_rewards_ep{episode}.pkl")
            with open(checkpoint_rewards_path, "wb") as f:
                pickle.dump(all_rewards[-50:], f) 

            print(f"[Checkpoint] Saved model and rewards at episode {episode}")

    final_model_path = os.path.join(save_dir, "dqn_trading_model_final.pth")
    agent.save(final_model_path, epsilon)
    with open(training_rewards_path, "wb") as f:
        pickle.dump(all_rewards, f)
    print(f"[Done] Training complete up to episode {start_episode + continue_episodes}")

    env.close()

if __name__ == "__main__":
    main()