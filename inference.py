import json
import math
import os
import random

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_inference(
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
        # Determine available question types in this row
        available = [qt for qt in question_types if pd.notna(row.get(qt))]
        # Sample M random question types if sample_qs > 0 and available > sample_qs; else use all
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
    out_dir = os.path.join("output", "inference")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "inference_results.json")
    with open(out_path, "w") as f:
        json.dump(outputs, f, indent=2)
    return outputs
