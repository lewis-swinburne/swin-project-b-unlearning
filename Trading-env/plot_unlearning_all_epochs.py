import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

results_dir = "unlearning_results"

benchmark_files = [f for f in os.listdir(results_dir) if f.startswith("benchmark_epochs") and f.endswith(".json")]

if not benchmark_files:
    raise FileNotFoundError(f"No benchmark files found in {results_dir}")

print(f"[Info] Found {len(benchmark_files)} benchmark files")

results = []
for filename in benchmark_files:
    filepath = os.path.join(results_dir, filename)
    with open(filepath, "r") as f:
        data = json.load(f)
        results.append({
            "epochs": data["unlearn_epochs"],
            "seed": data["seed"],
            "unlearn_time_minutes": data["unlearn_time_minutes"],
            "forget_mean_original": data["original_performance"]["forget_set_mean"],
            "forget_mean_unlearned": data["unlearned_performance"]["forget_set_mean"],
            "retain_mean_original": data["original_performance"]["retain_set_mean"],
            "retain_mean_unlearned": data["unlearned_performance"]["retain_set_mean"],
            "forget_degradation": data["metrics"]["forget_degradation_percent"],
            "retain_preservation": data["metrics"]["retain_preservation_percent"],
        })

df = pd.DataFrame(results)
df = df.sort_values(["epochs", "seed"])

print(f"[Info] Loaded data for epochs: {sorted(df['epochs'].unique())}")
print(f"[Info] Seeds used: {sorted(df['seed'].unique())}")

os.makedirs("exports", exist_ok=True)

epochs_list = sorted(df["epochs"].unique())
seeds_list = sorted(df["seed"].unique())

colors = plt.cm.tab10(np.linspace(0, 1, len(seeds_list)))
seed_colors = {seed: colors[i] for i, seed in enumerate(seeds_list)}

# ==============================================================================
# Figure 1: Forget Set Mean Reward (Original vs Unlearned)
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

for seed in seeds_list:
    seed_data = df[df["seed"] == seed].sort_values("epochs")
    ax.plot(seed_data["epochs"], seed_data["forget_mean_original"], 
            marker='o', linestyle='--', alpha=0.5, color=seed_colors[seed],
            label=f"Original (seed {seed})")
    ax.plot(seed_data["epochs"], seed_data["forget_mean_unlearned"], 
            marker='s', linestyle='-', linewidth=2, color=seed_colors[seed],
            label=f"Unlearned (seed {seed})")

ax.set_xlabel("Unlearning Epochs", fontsize=12)
ax.set_ylabel("Mean Reward", fontsize=12)
ax.set_title("Forget Set Performance: Original vs Unlearned Model", fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle=':', alpha=0.5, linewidth=1)
plt.tight_layout()
plt.savefig("exports/forget_set_comparison.png", dpi=200, bbox_inches='tight')
print("[Save] exports/forget_set_comparison.png")
plt.close()

# ==============================================================================
# Figure 2: Retain Set Mean Reward (Original vs Unlearned)
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

for seed in seeds_list:
    seed_data = df[df["seed"] == seed].sort_values("epochs")
    ax.plot(seed_data["epochs"], seed_data["retain_mean_original"], 
            marker='o', linestyle='--', alpha=0.5, color=seed_colors[seed],
            label=f"Original (seed {seed})")
    ax.plot(seed_data["epochs"], seed_data["retain_mean_unlearned"], 
            marker='s', linestyle='-', linewidth=2, color=seed_colors[seed],
            label=f"Unlearned (seed {seed})")

ax.set_xlabel("Unlearning Epochs", fontsize=12)
ax.set_ylabel("Mean Reward", fontsize=12)
ax.set_title("Retain Set Performance: Original vs Unlearned Model", fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle=':', alpha=0.5, linewidth=1)
plt.tight_layout()
plt.savefig("exports/retain_set_comparison.png", dpi=200, bbox_inches='tight')
print("[Save] exports/retain_set_comparison.png")
plt.close()

# ==============================================================================
# Figure 3: Unlearning Time vs Epochs
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

for seed in seeds_list:
    seed_data = df[df["seed"] == seed].sort_values("epochs")
    ax.plot(seed_data["epochs"], seed_data["unlearn_time_minutes"], 
            marker='o', linestyle='-', linewidth=2, color=seed_colors[seed],
            label=f"Seed {seed}")

ax.set_xlabel("Unlearning Epochs", fontsize=12)
ax.set_ylabel("Unlearning Time (minutes)", fontsize=12)
ax.set_title("Unlearning Time vs Number of Epochs", fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("exports/unlearning_time_comparison.png", dpi=200)
print("[Save] exports/unlearning_time_comparison.png")
plt.close()

# ==============================================================================
# Figure 4: Forget Degradation & Retain Preservation Percentages
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Forget Degradation
for seed in seeds_list:
    seed_data = df[df["seed"] == seed].sort_values("epochs")
    ax1.plot(seed_data["epochs"], seed_data["forget_degradation"], 
             marker='o', linestyle='-', linewidth=2, color=seed_colors[seed],
             label=f"Seed {seed}")

ax1.set_xlabel("Unlearning Epochs", fontsize=12)
ax1.set_ylabel("Forget Degradation (%)", fontsize=12)
ax1.set_title("Forget Set Degradation", fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='red', linestyle=':', alpha=0.5, linewidth=1)

# Retain Preservation
for seed in seeds_list:
    seed_data = df[df["seed"] == seed].sort_values("epochs")
    ax2.plot(seed_data["epochs"], seed_data["retain_preservation"], 
             marker='s', linestyle='-', linewidth=2, color=seed_colors[seed],
             label=f"Seed {seed}")

ax2.set_xlabel("Unlearning Epochs", fontsize=12)
ax2.set_ylabel("Retain Preservation (%)", fontsize=12)
ax2.set_title("Retain Set Preservation", fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(y=100, color='green', linestyle=':', alpha=0.5, linewidth=1, label='Perfect preservation')

plt.tight_layout()
plt.savefig("exports/degradation_preservation_comparison.png", dpi=200)
print("[Save] exports/degradation_preservation_comparison.png")
plt.close()

# End
print("\n" + "="*80)
print("ALL PLOTS SAVED SUCCESSFULLY")
print("="*80)