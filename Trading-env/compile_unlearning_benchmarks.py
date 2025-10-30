import os
import json
import csv

RESULTS_DIR = "unlearning_results"
OUTPUT_FILE = "exports/benchmark_summary.csv"

json_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("monitoring_epochs") and f.endswith(".json")]

if not json_files:
    print(f"No monitoring JSON files found in {RESULTS_DIR}/")
    exit(1)

rows = []

for filename in sorted(json_files):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "r") as f:
        data = json.load(f)
    
    epoch = data.get("unlearn_epochs", "unknown")
    seed = data.get("seed", "unknown")
    
    overall = data.get("overall_stats", {})
    
    row = {
        "unlearn_epochs": epoch,
        "seed": seed,
        "start_timestamp": overall.get("start_timestamp", "N/A"),
        "end_timestamp": overall.get("end_timestamp", "N/A"),
        "total_duration_seconds": overall.get("total_duration_seconds", "N/A"),
        "peak_cpu_percent": overall.get("peak_cpu_percent", "N/A"),
        "peak_memory_mb": overall.get("peak_memory_mb", "N/A"),
        "average_cpu_percent": overall.get("average_cpu_percent", "N/A"),
        "average_memory_mb": overall.get("average_memory_mb", "N/A")
    }

    rows.append(row)

with open(OUTPUT_FILE, "w", newline='') as out:
    if rows:
        fieldnames = rows[0].keys()
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

print(f"Performance summary written to {OUTPUT_FILE} ({len(rows)} runs collected)")