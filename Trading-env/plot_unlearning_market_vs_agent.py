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

df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl") # change this to ETHUSDT for ETH data

env = gym.make("TradingEnv",
    name="BTCUSD", # change this to ETHUSD for ETH data
    df=df,
    positions=[-1, 0, 1],
    trading_fees=0.01/100,
    borrow_interest_rate=0.0003/100,
)

from trading_learning import DQNAgent, ReplayBuffer
device = torch.device("cpu")

unlearned_model_path = "unlearning_results/unlearned_model_epochs75_seed1.pth" # Path to the unlearned model
if not os.path.exists(unlearned_model_path):
    raise FileNotFoundError(f"Unlearned model not found at {unlearned_model_path}")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim, ReplayBuffer(10000))

print(f"[Load] Loading unlearned model from {unlearned_model_path}")
checkpoint = torch.load(unlearned_model_path, map_location=device)
agent.model.load_state_dict(checkpoint["model"])
agent.target_model.load_state_dict(checkpoint["target_model"])
epsilon = checkpoint.get("epsilon", 0.05)
print(f"[Load] Model loaded successfully. Epsilon: {epsilon}")

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
plt.plot(norm_price, label="BTC Price (normalized)", color="gray", linewidth=1.5)
plt.plot(norm_portfolio, label="Unlearned Agent Portfolio Value (normalized)", color="dodgerblue", linewidth=2)
plt.title("Unlearned Agent Portfolio Value vs Market Price (Normalized % Change)")
plt.xlabel("Time (hours)")
plt.ylabel("Value (Starting = 100%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("exports/unlearned_agent_vs_market.png", dpi=200)
print("Plot saved to exports/unlearned_agent_vs_market.png")