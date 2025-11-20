import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from envs.cartpole_variants import make_env
import os

def evaluate(model, env, n_episodes=10):
    rewards = []
    records = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        rewards.append(total)
        records.append(total)
    return float(np.mean(rewards)), records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=str, default="runs/ppo_cart_ab.zip")
    parser.add_argument("--after", type=str, default="runs/ppo_cart_unlearnA.zip")
    parser.add_argument("--n_episodes", type=int, default=30)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    envA = make_env("A")
    envB = make_env("B")

    m_before = PPO.load(args.before, env=envB)
    m_after = PPO.load(args.after, env=envB)

    a_bef_mean, a_bef_recs = evaluate(m_before, envA, n_episodes=args.n_episodes)
    b_bef_mean, b_bef_recs = evaluate(m_before, envB, n_episodes=args.n_episodes)
    a_aft_mean, a_aft_recs = evaluate(m_after, envA, n_episodes=args.n_episodes)
    b_aft_mean, b_aft_recs = evaluate(m_after, envB, n_episodes=args.n_episodes)

    df = pd.DataFrame({
        "episode": list(range(args.n_episodes))*4,
        "env": (["A_before"]*args.n_episodes + ["B_before"]*args.n_episodes +
                ["A_after"]*args.n_episodes + ["B_after"]*args.n_episodes),
        "reward": a_bef_recs + b_bef_recs + a_aft_recs + b_aft_recs,
    })
    df.to_csv("results/eval_full.csv", index=False)

    summary = pd.DataFrame([
        {"env": "A", "stage": "before", "mean_reward": a_bef_mean},
        {"env": "B", "stage": "before", "mean_reward": b_bef_mean},
        {"env": "A", "stage": "after", "mean_reward": a_aft_mean},
        {"env": "B", "stage": "after", "mean_reward": b_aft_mean},
    ])
    summary.to_csv("results/summary.csv", index=False)

    print("\n=== RESULTS (avg over {} eps) ===".format(args.n_episodes))
    print(f"Env A (private) BEFORE: {a_bef_mean:.1f}")
    print(f"Env B (public)  BEFORE: {b_bef_mean:.1f}")
    print(f"Env A (private) AFTER : {a_aft_mean:.1f}")
    print(f"Env B (public)  AFTER : {b_aft_mean:.1f}\n")

if __name__ == "__main__":
    main()
