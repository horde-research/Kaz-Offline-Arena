import hashlib
import json
import math
import os
import tempfile
import uuid
import transformers
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from datasets import load_dataset
from typing import Literal
import logging

import openai
import pandas as pd
import torch  # noqa: F401
from dotenv import load_dotenv
from huggingface_hub.hf_api import HfFolder
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from tqdm import tqdm
import torch._dynamo
torch._dynamo.config.cache_size_limit = 1024  # or more


load_dotenv()
if "HUGGINGFACE_TOKEN" in os.environ:
    HfFolder.save_token(os.environ["HUGGINGFACE_TOKEN"])

# torch._dynamo.disable()
# torch._dynamo.config.cache_size_limit = 256


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sanitize_model_name(model_id: str) -> str:
    return model_id#.replace("/", "-")
def sanitize_model_name_postfix(model_id: str) -> str:
    return model_id.replace("/", "-")

def generate_postfix(indices: list, model_id: str, question_types: list, dt: datetime) -> str:
    indices_str = ",".join(sorted(str(x) for x in indices))
    base_str = f"{indices_str}_{model_id}_{'_'.join(question_types)}_{dt.strftime('%Y%m%d-%H%M%S')}"
    hash_val = hashlib.md5(base_str.encode()).hexdigest()[:8]
    return f"{hash_val}_{model_id}_{dt.strftime('%Y%m%d-%H%M%S')}"

def process_all_questions(df: pd.DataFrame, question_types: list):
    """
    Process all available questions from the DataFrame without sampling rows or question types.

    Parameters:
      df: pd.DataFrame
          The input data containing a "text" column for context and one or more question columns.
      question_types: list
          A list of column names representing different question types to be processed.

    Returns:
      mapping: list
          A list of dictionaries mapping task ids, question types, context, and corresponding prompts.
      prompts: list
          A list of prompt strings generated for every available question.
      unique_ids: list
          A list of unique identifiers for each processed (row, question_type) pair.
    """
    mapping = []
    prompts = []
    unique_ids = []

    # Process every row and every provided question type.
    for idx, row in df.iterrows():
        context = row["text"]
        for qt in question_types:
            if pd.notna(row.get(qt)):
                prompt = (
                    f"Context: {context}\n"
                    f"Question ({qt}): {row[qt]}\n\n"
                    "Answer the question in Kazakh language using the information provided in the context.\n"
                    "Be concise and clear—only answer the question asked, but answer it well."
                )
                prompts.append(prompt)
                mapping.append({
                    "task_id": idx,
                    "question": qt,
                    "question_type": qt,
                    "context": context,
                    "prompt": prompt,
                })
                unique_ids.append(f"{idx}-{qt}")
    return mapping, prompts, unique_ids

def save_results(final_results: dict, folder="inference",
                 file_name="inference_results.json"):
    out_dir = os.path.join("output", folder)
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile("w",
                                     dir=out_dir,
                                     delete=False,
                                     encoding="utf-8") as tmp:
        json.dump(final_results, tmp, indent=2, ensure_ascii=False)
        tmp_path = tmp.name
    # атомарный rename???
    final_path = os.path.join(out_dir, file_name)
    os.replace(tmp_path, final_path)
    print(f"Saved results to {final_path}")
    return final_path


def compute_token_stats(outputs: list):
    """
    Compute average tokens and standard deviation from the outputs.
    Each record is expected to have a "tokens_count" field.
    """
    token_counts = [rec.get("tokens_count", 0) for rec in outputs if rec.get("tokens_count") is not None]
    if token_counts:
        avg = sum(token_counts) / len(token_counts)
        std = math.sqrt(sum((tc - avg) ** 2 for tc in token_counts) / len(token_counts))
        return avg, std
    return 0, 0

def run_inference(
    model_id: str,
    question_types: list,
    batch_size: int,
    model_backend: Literal["hugginface", "openai"] = "hugginface",
    user_model_type: str = "SFT",  # user provided model type (default "SFT")
):
    if model_backend == "hugginface":
        outputs = run_inference_huggingface(model_id, question_types, batch_size)
    # elif model_backend == "openai":
    #     outputs = run_inference_openai(model_id, question_types, batch_size)
    else:
        raise ValueError(f"Unsupported model_backend: {model_backend}")

    dt = datetime.now()
    sampled_ids = [f"{rec['task_id']}-{rec['question_type']}" for rec in outputs]
    postfix = generate_postfix(sampled_ids, sanitize_model_name_postfix(model_id), question_types, dt)

    # Compute token statistics: average and standard deviation.
    avg_tokens, avg_tokens_std = compute_token_stats(outputs)

    # Create a final JSON structure that includes a summary and the detailed results.
    final_results = {
        "summary": {
            "avg_tokens": avg_tokens,
            "avg_tokens_std": avg_tokens_std,
            "model_type": user_model_type
        },
        "results": outputs
    }

    save_results(final_results, folder="inference", file_name=f"inference_results_{postfix}.json")
    return final_results

def run_inference_huggingface(
    model_id: str,
    question_types: list,
    batch_size: int
):
    print("Running Huggingface inference...")
    # Read full CSV data without sampling rows.
    
    # Load the dataset from Hugging Face with the "arena" split
    dataset = load_dataset("kz-transformers/arena-offline-qa", split="arena")

    # Convert to pandas DataFrame
    df = dataset.to_pandas()
    df = df.iloc[:2]  # Limit to 10 rows for testing
    # mapping, prompts, _ = process_all_questions(df, question_types)
    # print(f"Generated {len(prompts)} prompts for inference.")
# ——— INSERT ROUND-ROBIN QUESTION SELECTION HERE ———
    question_types = ['WHY_QS', 'WHAT_QS', 'HOW_QS', 'DESCRIBE_QS', 'ANALYZE_QS']
    n = len(question_types)
    # assign each row one question type in turn
    df['selected_qt'] = [question_types[i % n] for i in range(len(df))]
    df['selected_question'] = df.apply(lambda r: r[r['selected_qt']], axis=1)

    mapping, prompts, unique_ids = [], [], []
    for idx, row in df.iterrows():
        qt = row['selected_qt']
        qtext = row['selected_question']
        if pd.notna(qtext):
            prompt = (
                f"Context: {row['text']}\n"
                f"Question ({qt}): {qtext}\n\n"
                "Answer the question in Kazakh language using the information provided in the context.\n"
                "Be concise and clear—only answer the question asked, but answer it well."
            )
            mapping.append({
                "task_id": idx,
                "question_type": qt,
                "context": row['text'],
                "prompt": prompt
            })
            prompts.append(prompt)
            unique_ids.append(f"{idx}-{qt}")
    print(f"Generated {len(prompts)} prompts for inference (one per row).")

    if model_id == "inceptionai/Llama-3.1-Sherkala-8B-Chat":
        model_path = model_id
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Set chat template for Sherkala
        tokenizer.chat_template = (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"              
            "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
            "{{ content }}"
            "{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        )
        # Helper for generation
        def get_response(text):
            conversation = [{"role": "user", "content": text}]
            inputs = tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
            gen_tokens = model.generate(
                inputs,
                max_new_tokens=500,
                stop_strings=["<|eot_id|>"],
                tokenizer=tokenizer
            )
            # Decode excluding prompt tokens
            gen_text = tokenizer.decode(gen_tokens[0][len(inputs[0]):-1])
            return gen_text, gen_tokens, inputs

        # Run inference using get_response
        outputs = []
        for rec, prompt in zip(mapping, prompts):
            out_text, gen_tokens, inputs = get_response(prompt)
            token_count = gen_tokens[0].size(0) - inputs[0].size(0)
            rec.update({
                "output": out_text,
                "tokens_count": token_count,
                "model": sanitize_model_name(model_id),
                "generation_id": str(uuid.uuid4())
            })
            outputs.append(rec)

        print("Sherkala inference completed.")
        return outputs
    if model_id in ["google/gemma-3-4b-it","google/gemma-3-27b-it", "google/gemma-3-12b-it"]:
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True
        torch.set_float32_matmul_precision("high")
        torch._dynamo.config.cache_size_limit = 1024  # o

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16
        ).eval()
    
        outputs = []
        for rec in tqdm(mapping):
            messages = [
                {"role": "user",    "content": [
                    {"type": "text",  "text": rec["prompt"]}
                ]},
            ]
    
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors= "pt"
            ).to(model.device, dtype=torch.bfloat16)
            input_len = inputs["input_ids"].shape[-1]
    
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=500, do_sample=False)[0]
            gen = gen[input_len:]
    
            text = processor.decode(gen, skip_special_tokens=True)
            rec.update({
                "output": text,
                "tokens_count": gen.size(0),
                "model": sanitize_model_name(model_id),
                "generation_id": str(uuid.uuid4()),
            })
            outputs.append(rec)
    
        return outputs

    if model_id in ["Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]:

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        ).eval()

        outputs = []
        for rec in tqdm(mapping, desc="Qwen-32B inference"):

            messages = [
                {"role": "user", "content": rec["prompt"]},
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            gen_ids = model.generate(**model_inputs, max_new_tokens=512)

            stripped = [
                out_ids[input_ids.shape[-1]:]
                for input_ids, out_ids in zip(model_inputs.input_ids, gen_ids)
            ]

            response = tokenizer.batch_decode(stripped, skip_special_tokens=True)[0]

            rec.update({
                "output":       response,
                "tokens_count": stripped[0].size(0),
                "model":        sanitize_model_name(model_id),
                "generation_id": str(uuid.uuid4()),
            })
            outputs.append(rec)

        return outputs
    if model_id in ["meta-llama/Llama-3.3-70B-Instruct","meta-llama/Meta-Llama-3.1-8B-Instruct"]:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        pipe = transformers.pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tokenizer,
            device_map="auto",
            model_kwargs={"torch_dtype": torch.bfloat16},
        )

        outputs = []
        for rec in tqdm(mapping, desc="Llama-3.3-70B inference"):

            # build chat text with HF template
            chat_text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": rec["prompt"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

            generated = pipe(chat_text,
                             max_new_tokens=512,
                             do_sample=False)[0]["generated_text"]

            # strip the prompt part that we sent in
            answer = generated[len(chat_text):].strip()

            rec.update(
                {
                    "output":       answer,
                    "tokens_count": len(tokenizer(answer).input_ids),
                    "model":        sanitize_model_name(model_id),
                    "generation_id": str(uuid.uuid4()),
                }
            )
            outputs.append(rec)

        return outputs
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    extra = {}

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        **extra,
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

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
        "max_new_tokens": 512,
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
    for i in tqdm(range(num_batches)):
        batch_prompts = prompts[i * batch_size: (i + 1) * batch_size]
        logger.info(f"Processing batch {i+1}/{num_batches} with {len(batch_prompts)} prompts.")
        chat_inputs = [
            [{"role": "user", "content": prompt}] for prompt in batch_prompts
        ]

        # (Optionally set a chat template for specific models)
        chat_template = None
        if model_id in ["TilQazyna/llama-kaz-instruct-8B-1"]:
            chat_template = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"""
        elif model_id in ["IrbisAI/Irbis-7b-v0.1"]:
            chat_template = """{% for message in messages %}
            Сұрақ: {{ message['content'] | trim }}
            Жауап:
            {% endfor %}"""
        elif model_id in ["AmanMussa/llama2-kazakh-7b"]:
            chat_template = """"""

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
                input_ids=formatted_inputs,
                attention_mask=attention_masks,
                **generation_config,
            )
            if out_ids.ndim == 1:
                out_ids = out_ids.unsqueeze(0)
            #print("out_ids shape:", out_ids.shape)
        for j in range(len(batch_prompts)):
            input_length = formatted_inputs[j].shape[0]
            #print("out_ids j shape:", out_ids[j].shape)
            generated_tokens = out_ids[j][input_length:]
            out_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            if model_id not in [
                "TilQazyna/llama-kaz-instruct-8B-1",
                "IrbisAI/Irbis-7b-v0.1",
                "google/gemma-2-9b-it",
                "google/gemma-2-2b-it",
            ]:
                out_text = out_text[len("assistant"):].strip()
            elif model_id in ["Qwen/Qwen2.5-7B-Instruct"]:
                out_text = out_text[len("user\n\n"):].strip()
            rec = mapping[i * batch_size + j]
            rec["output"] = out_text
            rec["tokens_count"] = len(generated_tokens)
            rec["model"] = sanitize_model_name(model_id)
            rec["generation_id"] = str(uuid.uuid4())
            outputs.append(rec)
    logger.info("Inference completed.")
    return outputs

# def run_inference_openai(
#     model_id: str,
#     question_types: list,
#     batch_size: int,
# ):
#     # Load the dataset from Hugging Face with the "arena" split
#     dataset = load_dataset("kz-transformers/arena-offline-qa", split="arena")

#     # Convert to pandas DataFrame
#     df = dataset.to_pandas()
#     #df = pd.read_csv(tasks_csv)
#     mapping, prompts, _ = process_all_questions(df, question_types)
#     print(f"Generated {len(prompts)} prompts for inference.")

#     def call_openai(prompt_text: str) -> str:
#         messages = [{"role": "user", "content": prompt_text}]
#         response = openai.ChatCompletion.create(
#             model=model_id,
#             messages=messages,
#             max_tokens=512,
#             temperature=0.5,
#             top_p=0.75,
#         )
#         return response.choices[0].message.content.strip()

#     outputs = []
#     with ThreadPoolExecutor(max_workers=batch_size) as executor:
#         future_to_idx = {executor.submit(call_openai, p): idx for idx, p in enumerate(prompts)}
#         for future in as_completed(future_to_idx):
#             idx = future_to_idx[future]
#             try:
#                 out = future.result()
#             except Exception as e:
#                 out = f"Error: {e}"
#             rec = mapping[idx]
#             rec["output"] = out
#             rec["tokens_count"] = len(out.split())
#             rec["model"] = sanitize_model_name(model_id)
#             rec["unique_id"] = str(uuid.uuid4())
#             outputs.append(rec)
#     print("OpenAI chat inference completed.")
#     return outputs
