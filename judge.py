import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed


class JudgeOutput(BaseModel):
    explanation: str
    score: int


@retry(wait=wait_fixed(5), stop=stop_after_attempt(10))
def call_judge(prompt: str, model: str = "gpt-4") -> str:
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert evaluator. For the given context, question, and model response, "
                    "evaluate the quality of the response. Provide a chain-of-thought explanation and assign a score between 0 and 100. "
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
        f"Context: {record['context']}\n"
        f"Question ({record['question_type']}): Extracted from CSV.\n"
        f"Response: {record['output']}\n\n"
        "Evaluate the response and return a JSON object with keys 'explanation' (chain-of-thought) and 'score' (integer 0-100)."
    )
    try:
        resp = call_judge(prompt)
        judge_obj = JudgeOutput.parse_raw(resp)
        result = {
            "task_id": record["task_id"],
            "question_type": record["question_type"],
            "context": record["context"],
            "output": record["output"],
            "judge": judge_obj.dict(),
        }
    except Exception as e:
        result = {
            "task_id": record["task_id"],
            "question_type": record["question_type"],
            "context": record["context"],
            "output": record["output"],
            "judge": {"explanation": f"Error: {e}", "score": 0},
        }
    return result


def run_judgements():
    inf_path = os.path.join("output", "inference", "inference_results.json")
    with open(inf_path) as f:
        inference_results = json.load(f)
    judge_results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(judge_single, rec) for rec in inference_results]
        for future in as_completed(futures):
            judge_results.append(future.result())
    out_dir = os.path.join("output", "judge")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "judge_results.json")
    with open(out_path, "w") as f:
        json.dump(judge_results, f, indent=2)
    return judge_results
