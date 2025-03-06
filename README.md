# Offline Arena

This project runs a single Huggingface decoder-only model over tasks defined in a CSV. For each row, it randomly samples M question types (e.g. WHY_QS, WHAT_QS, etc.) and performs inference (one output per chosen question). Then, each row-question pair is sent individually to an LLM as judge. The judge returns a chain-of-thought explanation and a score (0–100) in JSON format (validated with Pydantic). Finally, pairwise ELO ratings are computed per row and aggregated into an Excel leaderboard.


## Setup

Install dependencies with Poetry:

```bash
poetry install
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
poetry run python src/main.py inference --model_id="huggingface/llama-3.1" --tasks_csv="tasks.csv" --sample_lines=50 --question_types="WHY_QS,WHAT_QS,HOW_QS,DESCRIBE_QS,ANALYZE_QS" --sample_qs=2 --batch_size=4
```

Run judge evaluations:
```bash
poetry run python src/main.py judge
```

Compute ELO leaderboard:
```bash
poetry run python src/main.py leaderboard
```
