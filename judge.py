import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from glob import glob

from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed


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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_retry(retry_state):
    logger.error(
        f"Retrying call_judge due to: {retry_state.outcome.exception()}. "
        f"Attempt {retry_state.attempt_number} of 10."
    )


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@retry(
    wait=wait_fixed(30) + wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(10),
    before_sleep=log_retry,
)
def call_judge(prompt: str, model: str = "gpt-4o") -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert evaluator with deep knowledge of Kazakh language, culture, history and context of Kazakhstan. "
                    "For the given context, question, and model response, evaluate the quality of the response. Provide an explanation "
                    "and assign a score between 0 and 10. 0 is completely irrelevant and unhelpful, 3 is partially relevant and helpful but incorrect, "
                    "5 is somewhat relevant and helpful but not fully correct, 7 is mostly relevant and helpful with some issues, "
                    "10 is completely relevant and helpful with no issues. Return a JSON object with keys 'explanation' and 'score'."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=3000,
    )
    return completion.choices[0].message.content


def clean_json_response(response: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        return match.group(1)
    return response.strip().replace("```json", "")


def judge_single(record: dict) -> dict:
    prompt = (
        f"**Context**: {record['context']}\n\n"
        f"**Question**: ({record['question_type']})\n\n"
        f"**Response**: {record['output']}\n\n"
        "Evaluate the response to the question based on context and return a JSON object with keys 'explanation' and 'score' (integer 0-10)."
    )
    try:
        resp = call_judge(prompt)
        judge_obj = JudgeOutput.parse_raw(clean_json_response(resp))
        result = {
            "task_id": record["task_id"],
            "question_type": record["question_type"],
            "context": record["context"],
            "output": record["output"],
            "model": record["model"],
            "generation_id": record["generation_id"],
            "judge": judge_obj.dict(),
            "success": True,
        }
    except Exception as e:
        result = {
            "task_id": record["task_id"],
            "question_type": record["question_type"],
            "context": record["context"],
            "output": record["output"],
            "model": record["model"],
            "generation_id": record["generation_id"],
            "judge": {"explanation": f"Error: {e}", "score": 0},
            "success": False,
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

    existing_judge_files = glob(os.path.join("output", "judge", "judge_results_*.json"))
    existing_judged_ids = set()
    for jf in existing_judge_files:
        print(f"Loading judged results from {jf}...")
        with open(jf) as f:
            jdata = json.load(f)
            for rec in jdata:
                if "generation_id" in rec and rec["success"]:
                    existing_judged_ids.add(rec["generation_id"])
    print(f"Found {len(existing_judged_ids)} already judged generation_ids.")

    to_judge = [
        rec
        for rec in inference_results
        if rec.get("generation_id") not in existing_judged_ids
    ]
    print(f"{len(to_judge)} inference records need to be judged.")

    judge_results = []
    indices = []
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = {executor.submit(judge_single, rec): rec for rec in to_judge}
        for i, future in enumerate(as_completed(futures)):
            if i % 100 == 0:
                print(f"Completed {i} tasks / {len(futures)}")
            res = future.result()
            judge_results.append(res)
            indices.append(f"{res['task_id']}-{res['question_type']}")

    all_judge_results = []
    for jf in existing_judge_files:
        with open(jf) as f:
            data = json.load(f)
            all_judge_results.extend(data)
    all_judge_results.extend(judge_results)

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
        json.dump(all_judge_results, f, indent=2)
    print(f"Judge evaluations completed. Results saved to {out_path}")
    return all_judge_results
