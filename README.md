# Offline Arena

This project runs a single Huggingface decoder-only model over tasks defined in a CSV. For each row, it randomly samples M question types (e.g. WHY_QS, WHAT_QS, etc.) and performs inference (one output per chosen question). Then, each row-question pair is sent individually to an LLM as judge. The judge returns a chain-of-thought explanation and a score (0–100) in JSON format (validated with Pydantic). Finally, pairwise ELO ratings are computed per row and aggregated into an Excel leaderboard.


## Setup

Install dependencies with Poetry:

```bash
poetry install
pip install flash-attn --no-build-isolation
```

## Env vars
Copy .env.template and fill in the required values


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
poetry run python main.py inference --model_id="meta-llama/Llama-3.2-3B-Instruct" --tasks_csv="Arena_QS_updated.csv" --sample_lines=50 --question_types="WHY_QS,WHAT_QS,HOW_QS,DESCRIBE_QS,ANALYZE_QS" --sample_qs=2 --batch_size=10
```

Run judge evaluations:
```bash
poetry run python main.py judge
```

Compute ELO leaderboard:
```bash
poetry run python main.py leaderboard
```
