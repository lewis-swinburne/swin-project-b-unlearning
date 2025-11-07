import os, argparse
from stable_baselines3 import PPO
from envs.cartpole_variants import make_env
from src.utils.monitor import ResourceLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_ckpt", type=str, default="runs/ppo_cart_ab")  # no .zip
    parser.add_argument("--finetune_steps", type=int, default=50_000)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--save", type=str, default="runs/ppo_cart_unlearnA")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tb_dir", type=str, default="tb/unlearn")
    args = parser.parse_args()

    os.makedirs("runs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    envB = make_env("B")

    model = PPO.load(args.from_ckpt, env=envB, tensorboard_log=args.tb_dir, seed=args.seed)
    model.lr_schedule = lambda _: args.lr

    print(f"🔁 Finetuning on B only for {args.finetune_steps} steps, lr={args.lr}")
    with ResourceLogger("results/usage_unlearn.csv"):
        model.learn(total_timesteps=args.finetune_steps, progress_bar=True, tb_log_name="ppo_cart_unlearn")

    model.save(f"{args.save}.zip")
    print(f"✅ Saved unlearned model to {args.save}.zip")

if __name__ == "__main__":
    main()
