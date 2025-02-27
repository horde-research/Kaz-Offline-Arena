import fire
from src import elo, inference, judge


def inference_cmd(
    model_id: str,
    tasks_csv: str = "tasks.csv",
    sample_lines: int = 100,
    question_types: str = "WHY_QS,WHAT_QS,HOW_QS,DESCRIBE_QS,ANALYZE_QS",
    sample_qs: int = 0,
    batch_size: int = 4,
):
    qtypes = [qt.strip() for qt in question_types.split(",") if qt.strip()]
    results = inference.run_inference(
        tasks_csv, model_id, sample_lines, qtypes, sample_qs, batch_size
    )
    print(
        "Inference completed. Results saved to output/inference/inference_results.json"
    )
    return results


def judge_cmd():
    results = judge.run_judgements()
    print(
        "Judge evaluations completed. Results saved to output/judge/judge_results.json"
    )
    return results


def elo_cmd():
    lb = elo.compute_elo()
    path = elo.save_leaderboard(lb)
    print("ELO leaderboard saved to", path)
    return lb.to_dict(orient="records")


if __name__ == "__main__":
    fire.Fire({"inference": inference_cmd, "judge": judge_cmd, "elo": elo_cmd})
