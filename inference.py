import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def save_results(
    outputs: list, folder: str = "inference", file_name: str = "inference_results.json"
):
    out_dir = os.path.join("output", folder)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, file_name)
    with open(out_path, "w") as f:
        json.dump(outputs, f, indent=2)
    return out_path


def run_inference(
    tasks_csv: str,
    model_id: str,
    sample_lines: int,
    question_types: list,
    sample_qs: int,
    batch_size: int,
    model_source: Literal["hugginface", "openai"] = "hugginface",
):
    if model_source == "hugginface":
        outputs = run_inference_huggingface(
            tasks_csv, model_id, sample_lines, question_types, sample_qs, batch_size
        )
    elif model_source == "openai":
        outputs = run_inference_openai(
            tasks_csv, model_id, sample_lines, question_types, sample_qs, batch_size
        )
    else:
        raise ValueError(f"Unsupported model_source: {model_source}")
    save_results(outputs)
    return outputs


def run_inference_huggingface(
    tasks_csv: str,
    model_id: str,
    sample_lines: int,
    question_types: list,
    sample_qs: int,
    batch_size: int,
):
    df = pd.read_csv(tasks_csv)
    if sample_lines and sample_lines < len(df):
        df = df.sample(n=sample_lines, random_state=42)
    prompts = []
    mapping = []
    for idx, row in df.iterrows():
        context = row["text"]
        available = [qt for qt in question_types if pd.notna(row.get(qt))]
        chosen = (
            random.sample(available, sample_qs)
            if sample_qs > 0 and len(available) > sample_qs
            else available
        )
        for qt in chosen:
            prompt = f"Context: {context}\nQuestion ({qt}): {row[qt]}"
            prompts.append(prompt)
            mapping.append(
                {
                    "task_id": idx,
                    "question_type": qt,
                    "context": context,
                    "prompt": prompt,
                }
            )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    outputs = []
    num_batches = math.ceil(len(prompts) / batch_size)
    for i in range(num_batches):
        batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
        encodings = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        encodings = {k: v.to(device) for k, v in encodings.items()}
        generated = model.generate(**encodings, max_new_tokens=256)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for j, out in enumerate(decoded):
            rec = mapping[i * batch_size + j]
            rec["output"] = out
            outputs.append(rec)
    return outputs


def run_inference_openai(
    tasks_csv: str,
    model_id: str,
    sample_lines: int,
    question_types: list,
    sample_qs: int,
    batch_size: int,
):
    df = pd.read_csv(tasks_csv)
    if sample_lines and sample_lines < len(df):
        df = df.sample(n=sample_lines, random_state=42)
    prompts = []
    mapping = []
    for idx, row in df.iterrows():
        context = row["text"]
        available = [qt for qt in question_types if pd.notna(row.get(qt))]
        chosen = (
            random.sample(available, sample_qs)
            if sample_qs > 0 and len(available) > sample_qs
            else available
        )
        for qt in chosen:
            prompt = f"Context: {context}\nQuestion ({qt}): {row[qt]}"
            prompts.append(prompt)
            mapping.append(
                {
                    "task_id": idx,
                    "question_type": qt,
                    "context": context,
                    "prompt": prompt,
                }
            )

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
            outputs.append(rec)
    return outputs
