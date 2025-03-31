import datetime
import json
import os
from collections import defaultdict
from glob import glob

import choix
import pandas as pd


def compute_elo():
    judge_files = glob(os.path.join("output", "judge", "judge_results_*.json"))
    if not judge_files:
        print("No judge result files found.")
        return pd.DataFrame()
    all_judge_results = []
    print(f"Found {len(judge_files)} judge file(s).")
    for file in judge_files:
        print(f"Loading judge results from {file}...")
        with open(file) as f:
            data = json.load(f)
            all_judge_results.extend(data)
    print(f"Total judge records loaded: {len(all_judge_results)}")

    # Group by generation_id (assume that records sharing the same generation_id correspond to the same input prompt)
    groups = defaultdict(list)
    for rec in all_judge_results:
        key = rec.get("task_id")
        if key is not None:
            groups[key].append(rec)

    comparisons = []  # List of (winner, loser) pairs by model name
    models_set = set()
    for group in groups.values():
        # Deduplicate within group: take the latest record per model.
        latest_by_model = {}
        for rec in group:
            model = rec.get("model")
            latest_by_model[model] = rec
        group_records = list(latest_by_model.values())

        if len(group_records) < 2:
            continue
        for i in range(len(group_records)):
            for j in range(i + 1, len(group_records)):
                score_i = group_records[i]["judge"].get("score", 0)
                score_j = group_records[j]["judge"].get("score", 0)
                model_i = group_records[i].get("model")
                model_j = group_records[j].get("model")
                models_set.add(model_i)
                models_set.add(model_j)
                if score_i > score_j:
                    comparisons.append((model_i, model_j))
                elif score_j > score_i:
                    comparisons.append((model_j, model_i))
                else:
                    # In a tie, count as a win for both directions.
                    comparisons.append((model_i, model_j))
                    comparisons.append((model_j, model_i))

    items = sorted(list(models_set))
    mapping = {item: idx for idx, item in enumerate(items)}
    comparisons_idx = [
        (mapping[winner], mapping[loser]) for winner, loser in comparisons
    ]

    n = len(items)
    print(
        f"Fitting Bradley–Terry model for {n} models using {len(comparisons_idx)} comparisons..."
    )
    bt_params = choix.ilsr_pairwise(n, comparisons_idx, alpha=0.01, max_iter=10000)

    leaderboard = pd.DataFrame({"model": items, "bt_strength": bt_params})
    leaderboard = leaderboard.sort_values("bt_strength", ascending=False).reset_index(
        drop=True
    )
    return leaderboard


def save_leaderboard(leaderboard):
    out_dir = os.path.join("output", "elo")
    os.makedirs(out_dir, exist_ok=True)

    dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_file = os.path.join(out_dir, f"leaderboard_{dt}.xlsx")
    leaderboard.to_excel(out_file, index=False)
    print(f"Saved leaderboard to {out_file}")
    return out_file
