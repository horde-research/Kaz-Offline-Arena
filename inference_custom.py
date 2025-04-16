import math
import uuid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_inference_custom(mapping, prompts, model_id, batch_size, random_seed=42):
    print("Running CUSTOM inference...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
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
            [{"role": "user", "content": prompt}] for prompt in batch_prompts
        ]
        # No model-specific chat template is used here.
        formatted_inputs = tokenizer.apply_chat_template(
            chat_inputs,
            tokenize=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
            chat_template=None,
        )
        formatted_inputs = formatted_inputs.to(device)
        attention_masks = []
        for input_ids in formatted_inputs:
            num_padding = 0
            for token_id in input_ids:
                if token_id == tokenizer.pad_token_id:
                    num_padding += 1
                else:
                    break
            attention_masks.append(
                [0] * num_padding + [1] * (len(input_ids) - num_padding)
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
        for j in range(len(batch_prompts)):
            input_length = formatted_inputs[j].shape[0]
            generated_tokens = out_ids[j][input_length:]
            out_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            rec = mapping[i * batch_size + j]
            rec["output"] = out_text
            rec["model"] = model_id
            rec["generation_id"] = str(uuid.uuid4())
            outputs.append(rec)
    print("Custom inference completed.")
    return outputs
