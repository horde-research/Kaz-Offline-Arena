from typing import Literal

import fire

import elo
from inference import run_inference
from judge import run_judgements


def inference_cmd(
    model_id: str,
    tasks_csv: str,
    sample_lines: int,
    question_types: str = "WHY_QS,WHAT_QS,HOW_QS,DESCRIBE_QS,ANALYZE_QS",
    sample_qs: int = 1,
    batch_size: int = 4,
    model_type: Literal["hugginface", "openai"] = "hugginface",
):
    print("Starting inference command...")
    if isinstance(question_types, str):
        question_types = question_types.split(",")
    else:
        question_types = list(question_types)
    qtypes = [qt.strip() for qt in question_types if qt.strip()]
    run_inference(
        tasks_csv, model_id, sample_lines, qtypes, sample_qs, batch_size, model_type
    )
    print("Inference completed.")


def judge_cmd():
    print("Starting judge evaluation command...")
    run_judgements()
    print("Judge evaluation completed.")


def elo_cmd():
    print("Starting ELO computation command...")
    lb = elo.compute_elo()
    if lb.empty:
        print("No leaderboard computed.")
        return {}
    path = elo.save_leaderboard(lb)
    print(f"ELO computation completed. Leaderboard saved to {path}")


if __name__ == "__main__":
    fire.Fire({"inference": inference_cmd, "judge": judge_cmd, "elo": elo_cmd})
