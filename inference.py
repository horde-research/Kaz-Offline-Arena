import hashlib
import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Literal

import openai
import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub.hf_api import HfFolder
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
if "HUGGINGFACE_TOKEN" in os.environ:
    HfFolder.save_token(os.environ["HUGGINGFACE_TOKEN"])


def sanitize_model_name(model_id: str) -> str:
    return model_id.replace("/", "-")


def generate_postfix(
    indices: list, model_id: str, sample_lines: int, sample_qs: int, dt: datetime
) -> str:
    indices_str = ",".join(sorted(str(x) for x in indices))
    base_str = f"{indices_str}_{model_id}_{sample_lines}_{sample_qs}_{dt.strftime('%Y%m%d-%H%M%S')}"
    hash_val = hashlib.md5(base_str.encode()).hexdigest()[:8]
    return f"{hash_val}_{dt.strftime('%Y%m%d-%H%M%S')}"


def sample_questions(
    df: pd.DataFrame, question_types: list, sample_qs: int, random_seed: int = 42
):
    random.seed(random_seed)
    mapping = []
    prompts = []
    sampled_ids = []
    for idx, row in df.iterrows():
        context = row["text"]
        available = [qt for qt in question_types if pd.notna(row.get(qt))]
        if not available:
            continue
        if sample_qs > 0 and len(available) > sample_qs:
            chosen = random.sample(available, sample_qs)
        else:
            chosen = available
        for qt in chosen:
            prompt = f"""
Context: {context}
Question ({qt}): {row[qt]}

Answer the question in Kazakh language, use information provided in the context.
"""
            prompts.append(prompt)
            mapping.append(
                {
                    "task_id": idx,
                    "question": qt,
                    "question_type": qt,
                    "context": context,
                    "prompt": prompt,
                }
            )
            sampled_ids.append(f"{idx}-{qt}")
    return mapping, prompts, sampled_ids


def save_results(
    outputs: list, folder: str = "inference", file_name: str = "inference_results.json"
):
    out_dir = os.path.join("output", folder)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, file_name)
    with open(out_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"Saved results to {out_path}")
    return out_path


def run_inference(
    tasks_csv: str,
    model_id: str,
    sample_lines: int,
    question_types: list,
    sample_qs: int,
    batch_size: int,
    model_type: Literal["hugginface", "openai"] = "hugginface",
    random_seed: int = 42,
):
    if model_type == "hugginface":
        outputs = run_inference_huggingface(
            tasks_csv,
            model_id,
            sample_lines,
            question_types,
            sample_qs,
            batch_size,
            random_seed,
        )
    elif model_type == "openai":
        outputs = run_inference_openai(
            tasks_csv,
            model_id,
            sample_lines,
            question_types,
            sample_qs,
            batch_size,
            random_seed,
        )
    else:
        raise ValueError(f"Unsupported model_source: {model_type}")
    dt = datetime.now()
    sampled_ids = [f"{rec['task_id']}-{rec['question_type']}" for rec in outputs]
    postfix = generate_postfix(
        sampled_ids, sanitize_model_name(model_id), sample_lines, sample_qs, dt
    )
    save_results(
        outputs, folder="inference", file_name=f"inference_results_{postfix}.json"
    )
    return outputs


def run_inference_huggingface(
    tasks_csv: str,
    model_id: str,
    sample_lines: int,
    question_types: list,
    sample_qs: int,
    batch_size: int,
    random_seed: int = 42,
):
    print("Running Huggingface inference...")
    df = pd.read_csv(tasks_csv)
    if sample_lines and sample_lines < len(df):
        df = df.sample(n=sample_lines, random_state=random_seed)
        print(f"Sampled {sample_lines} lines from {len(df)} total lines.")
    mapping, prompts, sampled_ids = sample_questions(
        df, question_types, sample_qs, random_seed
    )
    print(f"Generated {len(prompts)} prompts for inference.")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    outputs = []
    num_batches = math.ceil(len(prompts) / batch_size)
    print(f"Processing {num_batches} batches with batch size {batch_size}.")
    for i in range(num_batches):
        batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
        print(
            f"Processing batch {i+1}/{num_batches} with {len(batch_prompts)} prompts."
        )
        encodings = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        encodings = {k: v.to(device) for k, v in encodings.items()}
        with torch.no_grad():
            generated = model.generate(**encodings, max_new_tokens=256)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for j, out in enumerate(decoded):
            rec = mapping[i * batch_size + j]
            rec["output"] = out
            rec["model"] = sanitize_model_name(model_id)
            outputs.append(rec)
    print("Huggingface inference completed.")
    return outputs


def run_inference_openai(
    tasks_csv: str,
    model_id: str,
    sample_lines: int,
    question_types: list,
    sample_qs: int,
    batch_size: int,
    random_seed: int = 42,
):
    print("Running OpenAI inference...")
    df = pd.read_csv(tasks_csv)
    if sample_lines and sample_lines < len(df):
        df = df.sample(n=sample_lines, random_state=random_seed)
        print(f"Sampled {sample_lines} lines from {len(df)} total lines.")
    mapping, prompts, sampled_ids = sample_questions(
        df, question_types, sample_qs, random_seed
    )
    print(f"Generated {len(prompts)} prompts for inference.")

    def call_openai(prompt_text: str) -> str:
        response = openai.Completion.create(
            model=model_id,
            prompt=prompt_text,
            max_tokens=256,
            temperature=0.7,
        )
        return response.choices[0].text.strip()

    outputs = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        future_to_idx = {
            executor.submit(call_openai, p): idx for idx, p in enumerate(prompts)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                out = future.result()
            except Exception as e:
                out = f"Error: {e}"
            rec = mapping[idx]
            rec["output"] = out
            rec["model"] = sanitize_model_name(model_id)
            outputs.append(rec)
    print("OpenAI inference completed.")
    return outputs
