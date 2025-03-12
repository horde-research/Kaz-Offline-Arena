#!/bin/bash
models=(
  "meta/Lllam-3.1-8B"
  "meta/Lllam-3.1-70B"
)

for model in "${models[@]}"; do
  ts=$(date '+%Y%m%d_%H%M%S')
  safe_model=${model//\//_}
  log_file="${safe_model}_${ts}.log"
  echo "Running model: ${model}" | tee "${log_file}"
  original_batch=10
  batch_size=${original_batch}
  attempt=1
  while true; do
    echo "Attempt ${attempt} with batch_size ${batch_size}" | tee -a "${log_file}"
    cmd="poetry run python main.py inference --model_id=\"${model}\" --tasks_csv=\"Arena_QS_updated_filtered.csv\" --sample_lines=50 --question_types=\"WHY_QS,WHAT_QS,HOW_QS,DESCRIBE_QS,ANALYZE_QS\" --sample_qs=2 --batch_size=${batch_size}"
    echo "Command: ${cmd}" | tee -a "${log_file}"
    output=$(eval ${cmd} 2>&1 | tee -a "${log_file}")
    exit_code=${PIPESTATUS[0]}
    if [ ${exit_code} -eq 0 ]; then
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
