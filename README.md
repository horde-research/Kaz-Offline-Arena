# Offline Arena

This project runs a single Huggingface decoder-only model over tasks defined in a CSV. For each row, it randomly samples M question types (e.g. WHY_QS, WHAT_QS, etc.) and performs inference (one output per chosen question). Then, each row-question pair is sent individually to an LLM as judge. The judge returns a chain-of-thought explanation and a score (0–100) in JSON format (validated with Pydantic). Finally, pairwise ELO ratings are computed per row and aggregated into an Excel leaderboard.


## Setup

Install dependencies with Poetry:

```bash
poetry install
```

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
