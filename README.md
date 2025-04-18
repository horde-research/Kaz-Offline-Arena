# Offline Arena

This project runs a single Huggingface decoder-only model over tasks defined in a CSV. For each row, it randomly samples M question types (e.g. WHY_QS, WHAT_QS, etc.) and performs inference (one output per chosen question). Then, each row-question pair is sent individually to an LLM as judge. The judge returns a chain-of-thought explanation and a score (0–100) in JSON format (validated with Pydantic). Finally, pairwise ELO ratings are computed per row and aggregated into an Excel leaderboard.


## How to run inference with your model
The easiest way to run inference with your LLM is to re-implement `inference_custom.py` file.
The file contains a simple prototype of the inference, but you can modify it to suit your needs.
If model_id provided is unknown, inference_custom implementation will be used instead of the default one.

## Setup

Install dependencies with Poetry:

```bash
poetry install
poetry shell
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
Copy .env.template and fill in the required values


## Export requirements
```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
```


## Upload-download data
```bash
curl --progress-bar -F "file=@Arena_QS_updated.zip" https://0x0.st
```

Then copy response URL and download the file and unzip it
```bash
wget <your_public_url>
unzip <filename>.zip -d .
````


## Commands

Run inference:

```bash
poetry run python main.py inference --model_id="meta-llama/Llama-3.2-3B-Instruct" --tasks_csv="Arena_QS_updated_filtered.csv" --sample_lines=25 --question_types="WHY_QS,WHAT_QS,HOW_QS,DESCRIBE_QS,ANALYZE_QS" --sample_qs=2 --batch_size=50
```

Run judge evaluations:
```bash
poetry run python main.py judge
```

Compute ELO leaderboard:
## It's actually Bradley-Terry model
```bash
poetry run python main.py leaderboard
```
