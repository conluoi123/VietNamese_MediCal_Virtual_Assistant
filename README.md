# ViMedAQA — Vietnamese Medical Abstractive QA

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-ViMedAQA-red.svg)](https://huggingface.co/datasets/tmnam20/ViMedAQA)

> **Course:** Statistical Learning — HCMUS  
> **Task:** Abstractive Question Answering in the Vietnamese Medical Domain  
> **Dataset:** [tmnam20/ViMedAQA](https://huggingface.co/datasets/tmnam20/ViMedAQA) (ACL 2024)

---

## Overview

This project fine-tunes Vietnamese Transformer-based Seq2Seq models (**ViT5-base** and **BARTpho-word**) on the ViMedAQA dataset for abstractive medical question answering. A zero-shot **Llama-3.3-70B-versatile** baseline via Groq API is also included for comparison.

### RAG Pipeline (Retrieval-Augmented Generation)

The web application uses a **RAG architecture** powered by BM25 retrieval:

```
User Question → BM25 Retrieval (top-1 context) → ViT5 Generation → Answer
```

The end user only needs to input a **question** — the system automatically retrieves the most relevant medical context from a knowledge base of unique medical articles extracted from ViMedAQA, then feeds it to the fine-tuned model.

**Training Format (used during fine-tuning):**
```
INPUT : "question: {question} context: {context}"
OUTPUT: "{answer}"
```

---

## Project Structure

```
implementation/
├── data/
│   ├── raw/             # Full dataset backup (JSON)
│   ├── processed/       # train.json, val.json, test.json,
│   │                    # medical_corpus.json, bm25_index.pkl
│   └── eda/             # EDA charts and statistics CSV
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Phase 1 — EDA & data split
│   ├── 01b_build_retrieval_index.ipynb  # Phase 1.5 — Build BM25 knowledge base
│   ├── 02_baseline_groq.ipynb      # Phase 2 — Zero-shot baseline
│   ├── 03a_train_vit5.ipynb        # Phase 3A — ViT5-base fine-tuning
│   ├── 03b_train_bartpho.ipynb     # Phase 3B — BARTpho-word fine-tuning
│   ├── 03c_train_mt5.ipynb         # Phase 3C — mT5-base (optional)
│   ├── 04_evaluation.ipynb         # Phase 4 — Unified evaluation
│   └── 05_error_analysis.ipynb     # Phase 5 — Error analysis & ablation
├── src/
│   ├── data_utils.py    # Dataset loading, preprocessing, splitting
│   ├── metrics.py       # ROUGE, BLEU, BERTScore wrappers
│   ├── train_utils.py   # Trainer config, checkpoint logic
│   └── inference.py     # Single-sample and batch inference
├── app/
│   └── app.py           # Gradio web demo
├── report/
│   ├── main.tex
│   ├── references.bib
│   ├── sections/        # LaTeX section files
│   └── figures/         # Plots from EDA and evaluation
├── requirements.txt
└── .gitignore
```

---

## Models

| Model | Type | Params | HuggingFace ID |
|---|---|---|---|
| ViT5-base | Seq2Seq (T5) | ~270M | `VietAI/vit5-base` |
| BARTpho-word | Seq2Seq (BART) | ~396M | `vinai/bartpho-word` |
| mT5-base *(optional)* | Seq2Seq (T5) | ~580M | `google/mt5-base` |
| Llama-3.3-70B-versatile | LLM (zero-shot) | ~70B | Groq API |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Storage Strategy

- **Google Drive** — checkpoints, processed data, results (not committed to GitHub)
- **GitHub** — code and notebooks only (no model weights, no raw data)
- **HuggingFace Hub** — published fine-tuned model weights

---

## Citation

If you use ViMedAQA, please cite:

```bibtex
@inproceedings{tran-etal-2024-vimedaqa,
  title     = "{ViMedAQA}: A Vietnamese Medical Abstractive Question-Answering Dataset",
  author    = "Tran, Minh-Nam and Nguyen, Phu-Vinh and Nguyen, Long and Dinh, Dien",
  booktitle = "Proceedings of the 62nd Annual Meeting of ACL (Student Research Workshop)",
  year      = "2024",
  pages     = "252--260",
}
```
