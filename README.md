# Offline Arena

This project runs a single Huggingface decoder-only model over tasks defined in a CSV. For each row, it randomly samples M question types (e.g. WHY_QS, WHAT_QS, etc.) and performs inference (one output per chosen question). Then, each row-question pair is sent individually to an LLM as judge. The judge returns a chain-of-thought explanation and a score (0–100) in JSON format (validated with Pydantic). Finally, pairwise ELO ratings are computed per row and aggregated into an leaderboard.


## How to run inference with your model
The easiest way to run inference with your LLM is to re-implement `inference_custom.py` file.
The file contains a simple prototype of the inference, but you can modify it to suit your needs.
If model_id provided is unknown, inference_custom implementation will be used instead of the default one.

## install requirements
```bash
pip install -r requirements.txt
```

## Install Flash Attention
```bash
apt install gcc screen htop iotop nano
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run
#Accept terms
#Install only the CUDA Toolkit (no driver, since PyTorch works already)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
pip install --upgrade pip setuptools wheel
pip install flash-attn --no-build-isolation
python -m pip install --upgrade 'optree>=0.13.0'
```

## Env vars
Copy .env.template and fill in the required values in .env file

## Commands

Run inference:

```bash
python main.py inference --model_id="meta-llama/Llama-3.2-3B-Instruct" --question_types="WHY_QS,WHAT_QS,HOW_QS,DESCRIBE_QS,ANALYZE_QS" --batch_size=10
```

with nohup
```bash
nohup python main.py inference --model_id="issai/LLama-3.1-KazLLM-1.0-70B" --question_types="WHY_QS,WHAT_QS,HOW_QS,DESCRIBE_QS,ANALYZE_QS" --batch_size=1 > output.log 2>&1 &
```

Run judge evaluations:
```bash
python main.py judge
```

After Judge evaluations, you will get `filename.json` in the `output/judge/` directory. You can use this file to submit your model in the [Kaz Offline Arena](https://huggingface.co/spaces/kz-transformers/kaz-offline-arena).

## Note: You don't need to run the code(elo.py) below. We calculate elo in spaces leaderboard when you submit the model

Compute ELO leaderboard(optionally):
## It's actually Bradley-Terry model as described [here]()
```bash
python main.py elo
```
