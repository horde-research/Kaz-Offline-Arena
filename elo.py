import json
import os
from collections import defaultdict

import pandas as pd


def update_elo(rating_a, rating_b, outcome, k=32):
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1 - expected_a
    new_a = rating_a + k * (outcome - expected_a)
    new_b = rating_b + k * ((1 - outcome) - expected_b)
    return new_a, new_b


def compute_elo():
    judge_path = os.path.join("output", "judge", "judge_results.json")
    with open(judge_path) as f:
        judge_results = json.load(f)
    # Group by task_id so that each row with multiple QS evaluations is compared pairwise.
    grouped = {}
    for rec in judge_results:
        tid = rec["task_id"]
        if tid not in grouped:
            grouped[tid] = []
        grouped[tid].append(rec)
    ratings = defaultdict(lambda: 1500)
    # For each task with multiple QS outputs, compare pairwise based on score.
    for recs in grouped.values():
        if len(recs) < 2:
            continue
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                score_i = recs[i]["judge"].get("score", 0)
                score_j = recs[j]["judge"].get("score", 0)
                # if equal, consider it a draw
                if score_i == score_j:
                    outcome_i = 0.5
                elif score_i > score_j:
                    outcome_i = 1
                else:
                    outcome_i = 0
                # Update ratings pairwise
                qt_i = recs[i]["question_type"]
                qt_j = recs[j]["question_type"]
                r_i, r_j = ratings[qt_i], ratings[qt_j]
                new_r_i, new_r_j = update_elo(r_i, r_j, outcome_i)
                ratings[qt_i], ratings[qt_j] = new_r_i, new_r_j
    leaderboard = pd.DataFrame(
        [{"question_type": qt, "elo": int(r)} for qt, r in ratings.items()]
    )
    leaderboard = leaderboard.sort_values("elo", ascending=False).reset_index(drop=True)
    return leaderboard


def save_leaderboard(leaderboard):
    out_dir = os.path.join("output", "elo")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "leaderboard.xlsx")
    leaderboard.to_excel(out_file, index=False)
    return out_file
