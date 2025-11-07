import os, argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.cartpole_variants import make_env
from src.utils.monitor import ResourceLogger

def build_skewed_env_vec():
    return DummyVecEnv([
        lambda: make_env("B"),
        lambda: make_env("B"),
        lambda: make_env("B"),
        lambda: make_env("A"),
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--save", type=str, default="runs/ppo_cart_ab")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tb_dir", type=str, default="tb/train")
    args = parser.parse_args()

    os.makedirs("runs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    envs = build_skewed_env_vec()

    model = PPO(
        "MlpPolicy",
        envs,
        verbose=1,
        n_steps=1024,
        batch_size=1024,
        learning_rate=3e-4,
        gamma=0.99,
        n_epochs=10,
        seed=args.seed,
        tensorboard_log=args.tb_dir,
        policy_kwargs=dict(net_arch=[64, 64]),
    )

    with ResourceLogger("results/usage_train.csv"):
        model.learn(total_timesteps=args.steps, progress_bar=True, tb_log_name="ppo_cart_train")

    model.save(f"{args.save}.zip")
    print(f"✅ Saved model to {args.save}.zip")

if __name__ == "__main__":
    main()
