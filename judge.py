import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from glob import glob

import openai
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed


def sanitize_model_name(model_id: str) -> str:
    return model_id.replace("/", "-")


def generate_postfix(
    indices: list, model_name: str, sample_count: int, dt: datetime
) -> str:
    indices_str = ",".join(sorted(str(x) for x in indices))
    base_str = (
        f"{indices_str}_{model_name}_{sample_count}_{dt.strftime('%Y%m%d-%H%M%S')}"
    )
    hash_val = hashlib.md5(base_str.encode()).hexdigest()[:8]
    return f"{hash_val}_{dt.strftime('%Y%m%d-%H%M%S')}"


class JudgeOutput(BaseModel):
    explanation: str
    score: int


@retry(wait=wait_fixed(5), stop=stop_after_attempt(10))
def call_judge(prompt: str, model: str = "gpt-4o") -> str:
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert evaluator with deep knowledge of Kazakh language, culture, history and context of Kazakhstan. "
                    "For the given context, question, and model response, "
                    "evaluate the quality of the response. Provide an explanation and assign a score between 0 and 100. "
                    "Return a JSON object with keys 'explanation' and 'score'."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=3000,
    )
    return response.choices[0].message.content


def judge_single(record: dict) -> dict:
    prompt = (
        f"**Context**: {record['context']}\n\n"
        f"**Question**: ({record['question_type']})\n\n"
        f"**Response**: {record['output']}\n\n"
        "Evaluate the response to the question based on context and return a JSON object with keys 'explanation' and 'score' (integer 0-100)."
    )
    try:
        resp = call_judge(prompt)
        judge_obj = JudgeOutput.parse_raw(resp)
        result = {
            "task_id": record["task_id"],
            "question_type": record["question_type"],
            "context": record["context"],
            "output": record["output"],
            "model": record["model"],
            "judge": judge_obj.dict(),
        }
    except Exception as e:
        result = {
            "task_id": record["task_id"],
            "question_type": record["question_type"],
            "context": record["context"],
            "output": record["output"],
            "model": record["model"],
            "judge": {"explanation": f"Error: {e}", "score": 0},
        }
    return result


def run_judgements():
    inf_files = glob(os.path.join("output", "inference", "inference_results_*.json"))
    if not inf_files:
        print("No inference result files found.")
        return []
    print(f"Found {len(inf_files)} inference file(s).")
    inference_results = []
    for file in inf_files:
        print(f"Loading inference results from {file}...")
        with open(file) as f:
            data = json.load(f)
            inference_results.extend(data)
    print(f"Total inference records loaded: {len(inference_results)}")
    judge_results = []
    indices = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(judge_single, rec): rec for rec in inference_results}
        for future in as_completed(futures):
            res = future.result()
            judge_results.append(res)
            indices.append(f"{res['task_id']}-{res['question_type']}")
    dt = datetime.now()
    model_name = (
        inference_results[0].get("model", "unknown") if inference_results else "unknown"
    )
    postfix = generate_postfix(
        indices, sanitize_model_name(model_name), len(inference_results), dt
    )
    out_dir = os.path.join("output", "judge")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"judge_results_{postfix}.json")
    with open(out_path, "w") as f:
        json.dump(judge_results, f, indent=2)
    print(f"Judge evaluations completed. Results saved to {out_path}")
    return judge_results
