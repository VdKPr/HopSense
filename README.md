# HopSense — Multi-Hop Retrieval + Reasoning with Confidence

A RAG system that decomposes complex questions into atomic sub-questions, 
retrieves per hop, and synthesises with confidence propagation. Built to 
address failure modes of naive RAG on multi-hop questions.

## Why

Naive single-shot RAG collapses two-entity queries into one embedding 
and often misses one of the entities. On HotpotQA:

|  | Contains-gold | Latency |
|---|---|---|
| Naive RAG | 40% | 1.1s |
| **HopSense** | **58%** | 4.8s |

+18 pp accuracy at ~4× latency, with per-hop citations and confidence.

## Development log

The `docs/DEVLOG.md` file tracks every milestone: what was built, what 
failed, and why the next step was chosen. Problem files in 
`docs/problems/` document specific failure modes with reasoning and 
remediation options.

## Structure
HopSense/
├── src/
│ ├── decomposer.py # question → sub-questions
│ ├── retriever.py # FAISS + sentence-transformers
│ ├── pipeline.py # naive baseline + HopSense pipeline
│ ├── batch_eval.py # run on N questions
│ ├── analyze_results.py # per-type / calibration breakdown
│ └── load_data.py # HotpotQA loader
├── docs/
│ ├── DEVLOG.md
│ ├── outputs/ # every script's output, dated
│ └── problems/ # problem files (numbered by output)
├── results/ # JSON dumps of batch runs
└── data/ # HotpotQA sample (gitignored)


## Setup

```bash
python -m venv .venv_HopSense
.venv_HopSense\Scripts\activate      # Windows
pip install -r requirements.txt
python src/load_data.py              # downloads HotpotQA sample
```

Add `OPENAI_API_KEY=sk-...` to `.env`.

## Run

```bash
python src/pipeline.py                   # single question demo
python src/batch_eval.py                 # 10-100 question batch
python src/analyze_results.py            # breakdown of results
```

## Status

This is an active project. See `docs/DEVLOG.md` for latest milestone. 
Negative results are documented alongside positive ones.