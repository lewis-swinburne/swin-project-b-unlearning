import gymnasium as gym
import gym_trading_env
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend (no GUI)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt
import torch
import numpy as np
import os

df = pd.read_pickle("./data/binance-ETHUSDT-1h.pkl")

env = gym.make("TradingEnv",
    name="ETHUSD",
    df=df,
    positions=[-1, 0, 1],
    trading_fees=0.01/100,
    borrow_interest_rate=0.0003/100,
)

from trading_learning import DQNAgent, ReplayBuffer, get_latest_checkpoint
device = torch.device("cpu")

checkpoint_path = get_latest_checkpoint(directory="checkpoints")
if checkpoint_path is None:
    raise FileNotFoundError("No checkpoint found in the 'checkpoints' directory.")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim, ReplayBuffer(10000))
full_checkpoint_path = os.path.join("checkpoints", checkpoint_path)
epsilon = agent.load(full_checkpoint_path)

done, truncated = False, False
state, info = env.reset()
portfolio_values = []
prices = []
actions = []

while not done and not truncated:
    action = agent.get_action(state, epsilon=0.05)
    next_state, reward, done, truncated, info = env.step(action)
    state = next_state
    portfolio_values.append(info.get("portfolio_valuation", np.nan))
    prices.append(info.get("data_close", np.nan))
    actions.append(action)

env.close()

history_df = pd.DataFrame({
    "Price": prices,
    "Portfolio": portfolio_values,
    "Action": actions
}, index=df.index[:len(portfolio_values)])  

norm_price = history_df["Price"] / history_df["Price"].iloc[0] * 100
norm_portfolio = history_df["Portfolio"] / history_df["Portfolio"].iloc[0] * 100

plt.figure(figsize=(14, 6))
plt.plot(norm_price, label="ETH Price (normalized)", color="gray", linewidth=1.5)
plt.plot(norm_portfolio, label="Agent Portfolio Value (normalized)", color="dodgerblue", linewidth=2)
plt.title("Agent Portfolio Value vs ETH Market Price (Normalized % Change)")
plt.xlabel("Time (hours)")
plt.ylabel("Value (Starting = 100%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("exports/agent_vs_market_eth.png", dpi=200)
print("Plot saved to exports/agent_vs_market_eth.png")