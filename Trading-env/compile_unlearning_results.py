import os
import json
import csv

RESULTS_DIR = "unlearning_results"
OUTPUT_FILE = "exports/results_summary.csv"

json_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("benchmark_epochs") and f.endswith(".json")]

if not json_files:
    print(f"No benchmark JSON files found in {RESULTS_DIR}/")
    exit(1)

rows = []

for filename in sorted(json_files):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "r") as f:
        data = json.load(f)
    
    epoch = data.get("unlearn_epochs", "unknown")
    seed = data.get("seed", "unknown")
    
    forget_deg = data.get("metrics", {}).get("forget_degradation_percent", "N/A")
    retain_pres = data.get("metrics", {}).get("retain_preservation_percent", "N/A")
    
    orig_forget = data.get("original_performance", {}).get("forget_set_mean", "N/A")
    orig_retain = data.get("original_performance", {}).get("retain_set_mean", "N/A")
    unlearn_forget = data.get("unlearned_performance", {}).get("forget_set_mean", "N/A")
    unlearn_retain = data.get("unlearned_performance", {}).get("retain_set_mean", "N/A")

    row = {
        "epoch": epoch,
        "seed": seed,
        "forget_degradation_percent": forget_deg,
        "retain_preservation_percent": retain_pres,
        "original_forget_mean": orig_forget,
        "original_retain_mean": orig_retain,
        "unlearned_forget_mean": unlearn_forget,
        "unlearned_retain_mean": unlearn_retain
    }

    rows.append(row)

with open(OUTPUT_FILE, "w", newline='') as out:
    if rows:
        fieldnames = rows[0].keys()
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

print(f"Summary written to {OUTPUT_FILE} ({len(rows)} runs collected)")