import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_csv", type=str, default="results/eval_full.csv")
    parser.add_argument("--attack_csv", type=str, default="results/attack_population.csv")
    parser.add_argument("--out", type=str, default="results/benchmarks.png")
    args = parser.parse_args()

    df = pd.read_csv(args.eval_csv)

    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["A_before", "A_after", "B_before", "B_after"]
    means = [df[df.env == o].reward.mean() for o in order]
    ax.bar(order, means)
    ax.set_ylabel("Mean reward")
    ax.set_title("Unlearning benchmarks — mean rewards")
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved benchmark plot to {args.out}")

    agg = df.groupby("env")["reward"].agg(["mean", "std", "count"]).reset_index()
    agg.to_csv("results/benchmarks_summary.csv", index=False)

if __name__ == "__main__":
    main()
