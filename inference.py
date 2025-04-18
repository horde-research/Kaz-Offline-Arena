import hashlib
import json
import math
import os
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Literal

import openai
import pandas as pd
import peft
import torch  # noqa: F401
from dotenv import load_dotenv
from huggingface_hub.hf_api import HfFolder
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    return f"{hash_val}_{model_id}_{dt.strftime('%Y%m%d-%H%M%S')}"


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

Answer the question in Kazakh language, use information provided in the context. Be concise and clear, only answer the question asked, but answer it well.
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
        raise ValueError(f"Unsupported model_type: {model_type}")
    dt = datetime.now()
    sampled_ids = [f"{rec['task_id']}-{rec['question_type']}" for rec in outputs]
    postfix = generate_postfix(
        sampled_ids, sanitize_model_name(model_id), sample_lines, sample_qs, dt
    )
    save_results(
        outputs, folder="inference", file_name=f"inference_results_{postfix}.json"
    )
    return outputs


def prepare_data(
    tasks_csv: str,
    sample_lines: int,
    question_types: list,
    sample_qs: int,
    random_seed: int,
):
    df = pd.read_csv(tasks_csv)
    if sample_lines and sample_lines < len(df):
        df = df.sample(n=sample_lines, random_state=random_seed)
        print(f"Sampled {sample_lines} lines from {len(df)} total lines.")
    mapping, prompts, _ = sample_questions(df, question_types, sample_qs, random_seed)
    print(f"Generated {len(prompts)} prompts for inference.")
    return mapping, prompts


def run_inference_huggingface(
    tasks_csv: str,
    model_id: str,
    sample_lines: int,
    question_types: list,
    sample_qs: int,
    batch_size: int,
    random_seed: int = 42,
):
    mapping, prompts = prepare_data(
        tasks_csv, sample_lines, question_types, sample_qs, random_seed
    )
    SPECIAL_MODELS = [
        "armanibadboy/llama3.2-kazllm-3b-by-arman",
        "meta-llama/Llama-3.2-1B-Instruct",
        "TilQazyna/llama-kaz-instruct-8B-1",
        "google/gemma-2-2b-it",
        "AmanMussa/llama2-kazakh-7b",
        "IrbisAI/Irbis-7b-v0.1",
        "armanibadboy/llama3.1-kazllm-8b-by-arman-ver2",
        "meta-llama/Llama-3.2-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "google/gemma-2-9b-it",
        "issai/LLama-3.1-KazLLM-1.0-8B",
        "issai/LLama-3.1-KazLLM-1.0-70B",
    ]
    if model_id not in SPECIAL_MODELS:
        try:
            from inference_custom import run_inference_custom
        except ImportError as e:
            raise ImportError(
                "Custom inference module not found. Please provide inference_custom.py"
            ) from e
        return run_inference_custom(mapping, prompts, model_id, batch_size, random_seed)

    print("Running Huggingface inference...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # if model_id == "armanibadboy/llama3.2-kazllm-3b-by-arman":
    if False:  # Necessary on some machines, not on others, some dependency issue
        extra = {
            "gguf_file": "unsloth.Q8_0.gguf",
        }
    else:
        extra = {}
    if model_id == "armanibadboy/llama3.1-kazllm-8b-by-arman-ver2":
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = peft.PeftModel.from_pretrained(
            model, model_id, safe_serialization=True, torch_dtype=torch.bfloat16
        )
    elif model_id in ["issai/LLama-3.1-KazLLM-1.0-70B"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="cuda",
            trust_remote_code=True,
            **extra,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            **extra,
        )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if model_id in [
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
    ]:
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True
        torch.set_float32_matmul_precision("high")

    generation_config = {
        "do_sample": True,
        "max_new_tokens": 256,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "remove_invalid_values": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "forced_eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "no_repeat_ngram_size": 0,
        "num_return_sequences": 1,
    }

    outputs = []
    num_batches = math.ceil(len(prompts) / batch_size)
    print(f"Processing {num_batches} batches with batch size {batch_size}.")
    for i in range(num_batches):
        batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
        print(
            f"Processing batch {i+1}/{num_batches} with {len(batch_prompts)} prompts."
        )
        chat_inputs = [
            [
                {"role": "user", "content": prompt},
            ]
            for prompt in batch_prompts
        ]

        if model_id in [
            "TilQazyna/llama-kaz-instruct-8B-1",
        ]:
            chat_template = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"""
        elif model_id in [
            "IrbisAI/Irbis-7b-v0.1",
        ]:
            chat_template = """{% for message in messages %}
            Сұрақ: {{ message['content'] | trim }}
            Жауап:
            {% endfor %}"""
        elif model_id in [
            "AmanMussa/llama2-kazakh-7b",
        ]:
            chat_template = """"""
        else:
            chat_template = None
        formatted_inputs = tokenizer.apply_chat_template(
            chat_inputs,
            tokenize=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
            chat_template=chat_template,
        )
        formatted_inputs = formatted_inputs.to(device)
        attention_masks = []
        for input_ids in formatted_inputs:
            number_of_padding = 0
            for token_id in input_ids:
                if token_id == tokenizer.pad_token_id:
                    number_of_padding += 1
                else:
                    break
            attention_masks.append(
                [0] * number_of_padding + [1] * (len(input_ids) - number_of_padding)
            )
        attention_masks = torch.tensor(attention_masks).to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **{
                    "input_ids": formatted_inputs,
                    "attention_mask": attention_masks,
                },
                **generation_config,
            )
            if out_ids.ndim == 1:
                out_ids = out_ids.unsqueeze(0)
        for j in range(len(batch_prompts)):
            input_length = formatted_inputs[j].shape[0]
            generated_tokens = out_ids[j][input_length:]
            out_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            if model_id not in [
                "TilQazyna/llama-kaz-instruct-8B-1",
                "IrbisAI/Irbis-7b-v0.1",
                "google/gemma-2-9b-it",
                "google/gemma-2-2b-it",
            ]:
                out_text = out_text[len("assistant") :].strip()
            elif model_id in ["Qwen/Qwen2.5-7B-Instruct"]:
                out_text = out_text[len("user\n\n") :].strip()
            rec = mapping[i * batch_size + j]
            rec["output"] = out_text
            rec["model"] = sanitize_model_name(model_id)
            rec["generation_id"] = str(uuid.uuid4())
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
    print("Running OpenAI chat inference...")
    df = pd.read_csv(tasks_csv)
    if sample_lines and sample_lines < len(df):
        df = df.sample(n=sample_lines, random_state=random_seed)
        print(f"Sampled {sample_lines} lines from {len(df)} total lines.")
    mapping, prompts, sampled_ids = sample_questions(
        df, question_types, sample_qs, random_seed
    )
    print(f"Generated {len(prompts)} prompts for inference.")

    def call_openai(prompt_text: str) -> str:
        messages = [
            {"role": "user", "content": prompt_text},
        ]
        response = openai.ChatCompletion.create(
            model=model_id,
            messages=messages,
            max_tokens=256,
            temperature=0.5,
            top_p=0.75,
        )
        return response.choices[0].message.content.strip()

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
            rec["unique_id"] = str(uuid.uuid4())
            outputs.append(rec)
    print("OpenAI chat inference completed.")
    return outputs
