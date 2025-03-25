#!/bin/bash
mkdir -p ./logs

models=(
  "armanibadboy/llama3.2-kazllm-3b-by-arman"
  "meta-llama/Llama-3.2-1B-Instruct"
  "TilQazyna/llama-kaz-instruct-8B-1"
  "google/gemma-2-2b-it"
  "AmanMussa/llama2-kazakh-7b"
  "IrbisAI/Irbis-7b-v0.1"
  "armanibadboy/llama3.1-kazllm-8b-by-arman-ver2"
  "meta-llama/Llama-3.2-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
  "google/gemma-2-9b-it"
  "issai/LLama-3.1-KazLLM-1.0-8B"
)

for model in "${models[@]}"; do
  ts=$(date '+%Y%m%d_%H%M%S')
  safe_model=${model//\//_}
  log_file="./logs/${safe_model}_${ts}.log"
  echo "Running model: ${model}" | tee "${log_file}"
  original_batch=50
  batch_size=${original_batch}
  attempt=1
  while true; do
    echo "Attempt ${attempt} with batch_size ${batch_size}" | tee -a "${log_file}"
    cmd="python main.py inference --model_id=\"${model}\" --tasks_csv=\"Arena_QS_test_filtered_12k.csv\" --sample_lines=601 --question_types=\"WHY_QS,WHAT_QS,HOW_QS,DESCRIBE_QS,ANALYZE_QS\" --sample_qs=1 --batch_size=${batch_size}"
    echo "Command: ${cmd}" | tee -a "${log_file}"
    output=$(eval ${cmd} 2>&1 | tee -a "${log_file}")
    exit_code=${PIPESTATUS[0]}
    if [ ${exit_code} -eq 0 ] && ! echo "$output" | grep -q "ValueError:" ; then
      echo "Command succeeded on attempt ${attempt}" | tee -a "${log_file}"
      break
    fi
    if echo "$output" | grep -q "CUDA out of memory"; then
      if [ ${batch_size} -le 1 ]; then
        echo "Batch size reached minimum value. Exiting retry loop." | tee -a "${log_file}"
        break
      fi
      batch_size=$(( batch_size / 2 ))
      echo "CUDA out of memory detected. Reducing batch_size to ${batch_size} and retrying." | tee -a "${log_file}"
      attempt=$(( attempt + 1 ))
    else
      echo "Command failed with error not related to CUDA memory." | tee -a "${log_file}"
      break
    fi
  done
done
