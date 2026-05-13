# ViMedAQA — Vietnamese Medical Abstractive QA
## Complete Project Pipeline
**Course:** Statistical Learning — HCMUS  
**Dataset:** [tmnam20/ViMedAQA](https://huggingface.co/datasets/tmnam20/ViMedAQA) (ACL 2024)  
**Task:** Abstractive Question Answering in Vietnamese Medical Domain  
**Target:** 10/10 — production-quality, CV-ready, HuggingFace Hub publishable

---

## Table of Contents
1. [Project Architecture Decision](#0-architecture-decision)
2. [Phase 0 — Project Setup](#phase-0--project-setup)
3. [Phase 1 — Dataset Exploration & Preparation](#phase-1--dataset-exploration--preparation)
4. [Phase 1.5 — Build Medical Knowledge Base (RAG)](#phase-15--build-medical-knowledge-base-rag)
5. [Phase 2 — Zero-shot Baseline (Groq API)](#phase-2--zero-shot-baseline-groq-api)
6. [Phase 3A — Fine-tune ViT5-base](#phase-3a--fine-tune-vit5-base)
7. [Phase 3B — Fine-tune BARTpho-word](#phase-3b--fine-tune-bartpho-word)
8. [Phase 3C — Fine-tune mT5-base (Optional/Bonus)](#phase-3c--fine-tune-mt5-base-optionalbonus)
9. [Phase 4 — Unified Evaluation & Comparison](#phase-4--unified-evaluation--comparison)
10. [Phase 5 — Error Analysis & Ablation Study](#phase-5--error-analysis--ablation-study)
11. [Phase 6 — HuggingFace Hub Publication](#phase-6--huggingface-hub-publication)
12. [Phase 7 — Web Application (Gradio + RAG)](#phase-7--web-application-gradio--rag)
13. [Phase 8 — LaTeX Report](#phase-8--latex-report)
14. [Hardware & Account Strategy](#hardware--account-strategy)
15. [Git & Drive Strategy](#git--drive-strategy)
16. [Risk Register](#risk-register)

---

## 0. Architecture Decision

### Why Abstractive QA only (not Extractive)?
ViMedAQA is by design an **Abstractive QA** dataset — answers are not direct spans from context but paraphrased, synthesized responses. Forcing Extractive QA (span extraction like SQuAD) would violate the dataset's fundamental nature and produce poor results.

### Model Comparison Strategy

| Model | Type | Params | HF Hub ID | Rationale |
|---|---|---|---|---|
| **ViT5-base** | Seq2Seq (T5) | ~270M | `VietAI/vit5-base` | Vietnamese-specific, SOTA on Vi summarization |
| **BARTpho-word** | Seq2Seq (BART) | ~396M | `vinai/bartpho-word` | Vietnamese-specific, word-level tokenizer |
| **mT5-base** *(optional)* | Seq2Seq (T5) | ~580M | `google/mt5-base` | Multilingual baseline, shows cross-lingual benefit |
| **Llama-3.3-70B-versatile** | LLM (zero-shot) | ~70B | Groq API | Zero-shot open-weights baseline via Groq API |

**Primary deliverable:** ViT5-base + BARTpho-word fine-tuned and compared.  
**Bonus (if time):** mT5-base adds a third data point for ablation.

### Input/Output Format (unified across all Seq2Seq models)
```
INPUT : "question: {question} context: {context}"
OUTPUT: "{answer}"
```

---

## Phase 0 — Project Setup

### Description
Thiết lập toàn bộ hạ tầng dự án: GitHub repo, cấu trúc thư mục chuẩn, Google Drive layout, và môi trường Colab dùng chung. **Làm một lần duy nhất, mọi thành viên đều dùng chung cấu trúc này.**

### Environment
- **Hardware:** CPU only (không cần GPU)
- **Platform:** Local machine (git) + Google Drive (storage) + GitHub (code)

### Folder Structure (GitHub Repo)

```
vimedaq-transformer-qa/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_groq.ipynb
│   ├── 03a_train_vit5.ipynb
│   ├── 03b_train_bartpho.ipynb
│   ├── 03c_train_mt5.ipynb          # optional
│   ├── 04_evaluation.ipynb
│   └── 05_error_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py                # dataset loading, preprocessing, splitting
│   ├── metrics.py                   # ROUGE, BLEU, BERTScore wrappers
│   ├── train_utils.py               # Trainer config, checkpoint logic
│   └── inference.py                 # single-sample and batch inference
│
├── app/
│   └── app.py                       # Gradio web app
│
├── report/
│   ├── main.tex
│   ├── sections/
│   │   ├── introduction.tex
│   │   ├── dataset.tex
│   │   ├── methodology.tex
│   │   ├── experiments.tex
│   │   ├── results.tex
│   │   └── conclusion.tex
│   ├── figures/
│   └── references.bib
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Google Drive Layout

```
MyDrive/
└── vimedaq-project/
    ├── data/
    │   ├── raw/                     # downloaded from HuggingFace (parquet/json)
    │   ├── processed/
    │   │   ├── train.json
    │   │   ├── val.json
    │   │   └── test.json
    │   └── eda/                     # EDA charts, statistics CSV
    │
    ├── checkpoints/
    │   ├── vit5/
    │   │   ├── checkpoint-epoch-1/
    │   │   ├── checkpoint-epoch-2/
    │   │   └── best/
    │   ├── bartpho/
    │   │   ├── checkpoint-epoch-1/
    │   │   └── best/
    │   └── mt5/                     # optional
    │
    ├── results/
    │   ├── baseline_groq.json
    │   ├── baseline_groq_checkpoint.jsonl
    │   ├── eval_vit5.json
    │   ├── eval_bartpho.json
    │   ├── eval_mt5.json            # optional
    │   └── comparison_table.csv
    │
    └── logs/
        ├── vit5_training_log.csv
        └── bartpho_training_log.csv
```

### Tasks

1. **Create GitHub repo:** `vimedaq-transformer-qa` (public, MIT license)
2. **Create Google Drive folder:** `vimedaq-project/` — chia sẻ với tất cả thành viên (Editor access)
3. **Create `requirements.txt`:**
```
datasets==2.19.0
transformers==4.40.0
torch==2.2.0
evaluate==0.4.1
rouge-score==0.1.2
sacrebleu==2.3.1
bert-score==0.3.13
sentencepiece==0.2.0
gradio==4.31.0
groq
httpx<0.28.0
pandas==2.2.2
matplotlib==3.9.0
seaborn==0.13.2
numpy==1.26.4
tqdm==4.66.2
```
4. **Create `.gitignore`:** exclude `*.pt`, `*.bin`, `*.safetensors`, `data/`, `.env`
5. **Create `README.md`** template với badge shields

### Output
- GitHub repo initialized với full structure
- Google Drive folder shared với toàn nhóm
- `requirements.txt` committed

---

## Phase 1 — Dataset Exploration & Preparation

### Description
Load ViMedAQA, phân tích thống kê, làm sạch nếu cần, split train/val/test (stratified theo topic), lưu về Drive. **Chạy một lần duy nhất — các phase sau đều load từ Drive.**

### Environment
- **Hardware:** CPU (không cần GPU)
- **Platform:** Google Colab (bất kỳ account nào)
- **Runtime:** CPU runtime — tiết kiệm GPU quota
- **Estimated time:** ~20 phút

### Input
- `tmnam20/ViMedAQA` từ HuggingFace Hub

### Output
- `data/processed/train.json`, `val.json`, `test.json` → Google Drive
- `data/eda/stats.csv`, `data/eda/*.png` → Google Drive
- `data/raw/vimedaq_full.json` → Google Drive (backup)

### Notebook: `01_data_exploration.ipynb`

**Cell 1 — Mount Drive & Install**
```python
# Mount Google Drive (required every new session)
from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = "/content/drive/MyDrive/vimedaq-project"

# Install dependencies
!pip install datasets pandas matplotlib seaborn -q
```

**Cell 2 — Load Dataset**
```python
from datasets import load_dataset
import json, os

# Load from HuggingFace
dataset = load_dataset("tmnam20/ViMedAQA")
print(dataset)
print(dataset.column_names)

# Save raw copy to Drive immediately
os.makedirs(f"{DRIVE_ROOT}/data/raw", exist_ok=True)
# Convert to list of dicts and save
all_data = []
for split in dataset:
    for item in dataset[split]:
        item['_original_split'] = split
        all_data.append(item)

with open(f"{DRIVE_ROOT}/data/raw/vimedaq_full.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)
print(f"Raw data saved: {len(all_data)} samples")
```

> ⚠️ **NOTE:** Nếu dataset trên HF có sẵn train/test split, ghi chú lại để Phase 3 dùng đúng split. Nếu không có sẵn, dùng stratified split bên dưới.

**Cell 3 — EDA: Dataset Statistics**
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Combine all splits into one DataFrame
df = pd.DataFrame(all_data)
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("\nSample:")
print(df.head(2))

# Check for null values
print("\nNull counts:\n", df.isnull().sum())

# Topic distribution
print("\nTopic distribution:")
print(df['topic'].value_counts())

# Text length statistics
df['question_len'] = df['question'].apply(lambda x: len(x.split()))
df['context_len']  = df['context'].apply(lambda x: len(x.split()))
df['answer_len']   = df['answer'].apply(lambda x: len(x.split()))

print("\nText length stats (words):")
print(df[['question_len', 'context_len', 'answer_len']].describe())
```

**Cell 4 — EDA: Visualizations**
```python
import os
os.makedirs(f"{DRIVE_ROOT}/data/eda", exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Topic distribution
df['topic'].value_counts().plot(kind='bar', ax=axes[0,0], color='steelblue')
axes[0,0].set_title('Topic Distribution')
axes[0,0].set_xlabel('Topic')
axes[0,0].set_ylabel('Count')
axes[0,0].tick_params(axis='x', rotation=30)

# Question length distribution
axes[0,1].hist(df['question_len'], bins=30, color='coral', edgecolor='black')
axes[0,1].set_title('Question Length (words)')
axes[0,1].set_xlabel('Word count')

# Context length distribution
axes[1,0].hist(df['context_len'], bins=30, color='seagreen', edgecolor='black')
axes[1,0].set_title('Context Length (words)')
axes[1,0].set_xlabel('Word count')

# Answer length distribution
axes[1,1].hist(df['answer_len'], bins=30, color='purple', edgecolor='black')
axes[1,1].set_title('Answer Length (words)')
axes[1,1].set_xlabel('Word count')

plt.tight_layout()
plt.savefig(f"{DRIVE_ROOT}/data/eda/length_distributions.png", dpi=150, bbox_inches='tight')
plt.show()
print("EDA charts saved to Drive.")
```

**Cell 5 — Train/Val/Test Split**
```python
from sklearn.model_selection import train_test_split

# Stratified split by topic: 80% train, 10% val, 10% test
df_clean = df.dropna(subset=['question', 'context', 'answer', 'topic'])
print(f"Clean samples: {len(df_clean)}")

# If HF already provides splits, use those and skip this cell
# Check if '_original_split' has train/test/validation
print(df_clean['_original_split'].value_counts())
```

```python
# If no pre-defined split exists (only 'train' split in HF):
# Stratified split
train_df, temp_df = train_test_split(
    df_clean, test_size=0.2, random_state=42, stratify=df_clean['topic']
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df['topic']
)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print("\nTrain topic distribution:\n", train_df['topic'].value_counts())
print("\nTest topic distribution:\n", test_df['topic'].value_counts())

# Save splits
os.makedirs(f"{DRIVE_ROOT}/data/processed", exist_ok=True)
keep_cols = ['question', 'context', 'answer', 'topic']

train_df[keep_cols].to_json(f"{DRIVE_ROOT}/data/processed/train.json",
                             orient='records', force_ascii=False, indent=2)
val_df[keep_cols].to_json(f"{DRIVE_ROOT}/data/processed/val.json",
                           orient='records', force_ascii=False, indent=2)
test_df[keep_cols].to_json(f"{DRIVE_ROOT}/data/processed/test.json",
                            orient='records', force_ascii=False, indent=2)
print("Splits saved to Drive successfully.")
```

**Cell 6 — Save Stats CSV**
```python
stats = {
    'total_samples': len(df_clean),
    'train_size': len(train_df),
    'val_size': len(val_df),
    'test_size': len(test_df),
    'num_topics': df_clean['topic'].nunique(),
    'topics': df_clean['topic'].unique().tolist(),
    'avg_question_len': round(df_clean['question_len'].mean(), 2),
    'avg_context_len': round(df_clean['context_len'].mean(), 2),
    'avg_answer_len': round(df_clean['answer_len'].mean(), 2),
    'max_context_len': int(df_clean['context_len'].max()),
}
pd.DataFrame([stats]).to_csv(f"{DRIVE_ROOT}/data/eda/dataset_stats.csv", index=False)
print("Stats saved:", stats)
```

### ✅ Checklist Phase 1
- [ ] `vimedaq_full.json` saved to Drive
- [ ] `train.json`, `val.json`, `test.json` saved to Drive
- [ ] `length_distributions.png`, `dataset_stats.csv` saved to Drive
- [ ] Note down: total samples, max context length (important for tokenizer max_length in Phase 3)

---

## Phase 1.5 — Build Medical Knowledge Base (RAG)

### Description
Xây dựng kho tri thức y khoa (knowledge base) từ toàn bộ context duy nhất trong dataset. Dùng BM25Okapi để tạo chỉ mục truy xuất. **Phục vụ Phase 7 (Web App) — cho phép user chỉ cần nhập câu hỏi, không cần cung cấp context.**

> ⚠️ **KHÔNG ảnh hưởng đến training.** BM25 retrieval chỉ dùng ở inference time (Web App). Training vẫn dùng gold context.

### Environment
- **Hardware:** CPU only
- **Platform:** Google Colab (CPU runtime) hoặc local
- **Estimated time:** < 5 phút

### Input
- `data/raw/vimedaq_full.json` (từ Phase 1)

### Output
- `data/processed/medical_corpus.json` — danh sách context duy nhất (sorted)
- `data/processed/bm25_index.pkl` — BM25Okapi index đã serialize

### Notebook: `01b_build_retrieval_index.ipynb`

**Workflow:**
1. Load `vimedaq_full.json`
2. Trích xuất tất cả context duy nhất (deduplicate) → sorted list
3. Tokenize bằng whitespace splitting → xây BM25Okapi index
4. Lưu corpus (`.json`) và BM25 index (`.pkl`) vào `data/processed/`
5. Validation: test retrieval với 3 câu hỏi mẫu, kiểm tra reload

### ✅ Checklist Phase 1.5
- [ ] `medical_corpus.json` saved (kiểm tra số lượng documents hợp lý)
- [ ] `bm25_index.pkl` saved (kiểm tra reload thành công)
- [ ] Retrieval validation: top-1 context có liên quan đến câu hỏi
- [ ] KHÔNG thay đổi `train.json`, `val.json`, `test.json`

---

## Phase 2 — Zero-shot Baseline (Groq API)

### Description
Dùng **Llama-3.3-70B-versatile** (zero-shot, không fine-tune) qua Groq API để trả lời câu hỏi trên **test set**. Đây là external baseline chứng minh fine-tuning của nhóm có ý nghĩa.

### Environment
- **Hardware:** CPU
- **Platform:** Google Colab (CPU runtime)
- **API Key:** Lấy từ Groq Console (miễn phí)
- **API Limits:** 30 requests/phút, 100,000 tokens/ngày (TPD)
- **Strategy:** Sử dụng cơ chế Checkpointing và I/O append mode (ghi tệp `.jsonl`) như một quy chuẩn bắt buộc để tránh mất dữ liệu khi đạt giới hạn. Thiết lập luân phiên nhiều API Key nếu cần.

### Input
- `data/processed/test.json` từ Drive

### Output
- `results/baseline_groq_checkpoint.jsonl` → Drive (lưu vết)
- `results/baseline_groq.json` → Drive (câu trả lời dự đoán + metrics)

### Notebook: `02_baseline_groq.ipynb`

**Cell 1 — Setup**
```python
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = "/content/drive/MyDrive/vimedaq-project"

!pip install google-generativeai evaluate rouge-score bert-score sacrebleu -q
```

**Cell 2 — Verify API Model is Active**
```python
import google.generativeai as genai
import os

# Set your API key (get from https://aistudio.google.com/apikey)
GEMINI_API_KEY = "YOUR_API_KEY_HERE"  # Replace this
genai.configure(api_key=GEMINI_API_KEY)

# Verify model availability
MODEL_NAME = "gemini-2.5-flash"  # stable as of April 2026
try:
    model = genai.GenerativeModel(MODEL_NAME)
    test_response = model.generate_content("Hello, respond with 'OK'")
    print(f"✅ Model {MODEL_NAME} is active: {test_response.text}")
except Exception as e:
    print(f"❌ Model error: {e}")
    print("Check https://ai.google.dev/gemini-api/docs/models for current model names")
```

**Cell 3 — Load Test Data**
```python
import json
import pandas as pd

with open(f"{DRIVE_ROOT}/data/processed/test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

test_df = pd.DataFrame(test_data)
print(f"Test samples: {len(test_df)}")
print(test_df.head(2))
```

**Cell 4 — Zero-shot Inference**
```python
import time
from tqdm import tqdm

def generate_answer_zero_shot(question: str, context: str, model) -> str:
    """Zero-shot prompt for Vietnamese medical QA."""
    prompt = f"""Bạn là trợ lý y tế chuyên nghiệp. Dựa vào thông tin được cung cấp, hãy trả lời câu hỏi một cách chính xác và ngắn gọn bằng tiếng Việt.

Context: {context}

Question: {question}

Answer:"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error: {e}")
        return ""

# Run inference with rate limiting
predictions = []
references  = []
failed_indices = []

for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Gemini inference"):
    pred = generate_answer_zero_shot(row['question'], row['context'], model)
    if pred:
        predictions.append(pred)
        references.append(row['answer'])
    else:
        failed_indices.append(i)
        predictions.append("")
        references.append(row['answer'])
    time.sleep(1.0)  # Rate limit: 1 request/sec for free tier

print(f"\nDone. Failed: {len(failed_indices)}")
```

**Cell 5 — Compute Metrics**
```python
import evaluate

rouge = evaluate.load("rouge")
bleu  = evaluate.load("bleu")

# Filter out empty predictions
valid_pairs = [(p, r) for p, r in zip(predictions, references) if p]
preds_valid = [p for p, r in valid_pairs]
refs_valid  = [r for p, r in valid_pairs]

# ROUGE
rouge_scores = rouge.compute(predictions=preds_valid, references=refs_valid)
print("ROUGE scores:", {k: round(v, 4) for k, v in rouge_scores.items()})

# BLEU
bleu_score = bleu.compute(predictions=preds_valid,
                           references=[[r] for r in refs_valid])
print(f"BLEU-4: {round(bleu_score['bleu'], 4)}")
```

**Cell 6 — Save Results to Drive**
```python
import os
os.makedirs(f"{DRIVE_ROOT}/results", exist_ok=True)

baseline_results = {
    "model": MODEL_NAME,
    "mode": "zero-shot",
    "num_samples": len(preds_valid),
    "rouge1": round(rouge_scores['rouge1'], 4),
    "rouge2": round(rouge_scores['rouge2'], 4),
    "rougeL": round(rouge_scores['rougeL'], 4),
    "bleu4":  round(bleu_score['bleu'], 4),
    "predictions": [{"question": test_df.iloc[i]['question'],
                     "reference": references[i],
                     "prediction": predictions[i]}
                    for i in range(len(predictions))]
}

with open(f"{DRIVE_ROOT}/results/baseline_gemini.json", "w", encoding="utf-8") as f:
    json.dump(baseline_results, f, ensure_ascii=False, indent=2)
print("Baseline results saved to Drive.")
```

### ✅ Checklist Phase 2
- [ ] Llama-3.3-70B-versatile responds correctly
- [ ] `baseline_groq.json` and `baseline_groq_checkpoint.jsonl` saved to Drive with ROUGE/BLEU scores
- [ ] Note down ROUGE-L và BLEU-4 scores (will be the "ceiling" to compare against)

---

## Phase 3A — Fine-tune ViT5-base

### Description
Fine-tune **ViT5-base** (`VietAI/vit5-base`, ~270M params) trên ViMedAQA train set. **Đây là model chính, ưu tiên cao nhất.**

### Environment
- **Hardware:** T4 GPU (16GB VRAM) — **BẮT BUỘC phải chọn T4 GPU runtime**
- **Platform:** Google Colab (bất kỳ account nào trong 5 accounts)
- **RAM needed:** ~8-10GB RAM + ~6GB VRAM
- **Estimated training time:** ~45-90 phút/epoch (tùy dataset size) — **⚠️ > 15 phút, PHẢI có checkpoint**
- **Checkpoint saved to Drive:** sau mỗi epoch và best model

> ⚠️ **COLAB SESSION WARNING:** Colab Free disconnect sau ~12 giờ idle. Luôn lưu checkpoint vào Drive. Nếu session reset, chạy lại từ Cell 1 và resume từ checkpoint.

### Input
- `data/processed/train.json`, `val.json` từ Drive

### Output
- `checkpoints/vit5/best/` → Drive
- `logs/vit5_training_log.csv` → Drive
- `results/eval_vit5.json` → Drive

### Notebook: `03a_train_vit5.ipynb`

**Cell 1 — Verify GPU & Mount Drive**
```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    raise RuntimeError("⚠️ NO GPU DETECTED. Go to Runtime > Change runtime type > T4 GPU. Then re-run from Cell 1.")

from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = "/content/drive/MyDrive/vimedaq-project"

import os
os.makedirs(f"{DRIVE_ROOT}/checkpoints/vit5/best", exist_ok=True)
os.makedirs(f"{DRIVE_ROOT}/logs", exist_ok=True)
```

**Cell 2 — Install Dependencies**
```python
!pip install transformers==4.40.0 datasets evaluate rouge-score sacrebleu sentencepiece accelerate -q
print("Installation complete.")
```

**Cell 3 — Load Data**
```python
import json
import pandas as pd
from datasets import Dataset

def load_split(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

train_data = load_split(f"{DRIVE_ROOT}/data/processed/train.json")
val_data   = load_split(f"{DRIVE_ROOT}/data/processed/val.json")

train_dataset = Dataset.from_list(train_data)
val_dataset   = Dataset.from_list(val_data)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
print("Sample:", train_dataset[0])
```

**Cell 4 — Tokenizer & Preprocessing**
```python
from transformers import AutoTokenizer

MODEL_NAME      = "VietAI/vit5-base"
MAX_INPUT_LEN   = 512   # question + context
MAX_TARGET_LEN  = 128   # answer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(examples):
    # T5 input format: "question: {q} context: {c}"
    inputs = [
        f"question: {q} context: {c}"
        for q, c in zip(examples['question'], examples['context'])
    ]
    targets = examples['answer']

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LEN,
        padding="max_length",
        truncation=True,
    )
    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LEN,
        padding="max_length",
        truncation=True,
    )
    # Replace padding token id in labels with -100 (ignore in loss)
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_tokenized = train_dataset.map(preprocess, batched=True,
                                     remove_columns=train_dataset.column_names)
val_tokenized   = val_dataset.map(preprocess, batched=True,
                                   remove_columns=val_dataset.column_names)
print("Tokenization complete.")
print("Train tokenized shape:", train_tokenized.shape)
```

**Cell 5 — Load Model**
```python
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model = model.to('cuda')

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params/1e6:.1f}M | Trainable: {trainable_params/1e6:.1f}M")

# VRAM check
print(f"Model VRAM usage: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

**Cell 6 — Training Configuration**
```python
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import evaluate
import numpy as np

# OOM-safe config for T4 (16GB VRAM, ~270M param model)
BATCH_SIZE          = 4     # per device
GRAD_ACCUM          = 4     # effective batch = 4 * 4 = 16
LEARNING_RATE       = 3e-4
NUM_EPOCHS          = 5
WARMUP_STEPS        = 100
SAVE_STEPS          = 100   # save checkpoint every 100 steps
EVAL_STEPS          = 100

CHECKPOINT_DIR      = "/content/vit5_checkpoints"  # local Colab storage
BEST_MODEL_DIR      = f"{DRIVE_ROOT}/checkpoints/vit5/best"

training_args = Seq2SeqTrainingArguments(
    output_dir=CHECKPOINT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    evaluation_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=2,           # keep only 2 local checkpoints (save disk)
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    fp16=True,                    # mixed precision for T4
    logging_dir=f"{DRIVE_ROOT}/logs",
    logging_steps=50,
    report_to="none",             # no WandB needed
    dataloader_num_workers=2,
)
print("Training args configured.")
```

**Cell 7 — Metrics Function**
```python
rouge_metric = evaluate.load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Strip whitespace
    decoded_preds  = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]
    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {k: round(v, 4) for k, v in result.items()}
```

**Cell 8 — Train**
```python
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ⚠️ ESTIMATED TIME: 45-90 min/epoch × 5 epochs = 3-7 hours total
# Session may disconnect — checkpoints saved to local /content/, copy to Drive below
print("Starting training... This will take ~3-7 hours.")
print("⚠️ Do NOT close browser. Enable 'Keep awake' extension if possible.")

trainer.train()
print("Training complete!")
```

**Cell 9 — Save Best Model to Drive (CRITICAL)**
```python
import shutil

# Save best model to Drive
trainer.save_model(BEST_MODEL_DIR)
tokenizer.save_pretrained(BEST_MODEL_DIR)
print(f"✅ Best model saved to Drive: {BEST_MODEL_DIR}")

# Save training log
log_history = trainer.state.log_history
import pandas as pd
pd.DataFrame(log_history).to_csv(f"{DRIVE_ROOT}/logs/vit5_training_log.csv", index=False)
print("Training log saved to Drive.")
```

**Cell 10 — Quick Validation Inference Check**
```python
# Sanity check: run inference on 3 val samples
model.eval()
sample = val_data[:3]

for item in sample:
    input_text = f"question: {item['question']} context: {item['context']}"
    inputs = tokenizer(input_text, return_tensors="pt",
                       max_length=MAX_INPUT_LEN, truncation=True).to('cuda')
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=MAX_TARGET_LEN,
                                  num_beams=4, early_stopping=True)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Q: {item['question'][:80]}")
    print(f"Pred: {pred}")
    print(f"Ref : {item['answer'][:100]}")
    print("---")
```

> ⚠️ **RESUME AFTER SESSION RESET:**
> Nếu Colab disconnect giữa chừng, chạy lại Cell 1-3, sau đó:
> ```python
> trainer.train(resume_from_checkpoint=CHECKPOINT_DIR + "/checkpoint-XXXX")
> ```

### ✅ Checklist Phase 3A
- [ ] T4 GPU runtime confirmed (Cell 1 passes)
- [ ] Training completed (hoặc resume từ checkpoint)
- [ ] Best model saved to `checkpoints/vit5/best/` on Drive
- [ ] `vit5_training_log.csv` saved to Drive
- [ ] Sanity check inference produces valid Vietnamese text

---

## Phase 3B — Fine-tune BARTpho-word

### Description
Fine-tune **BARTpho-word** (`vinai/bartpho-word`, ~396M params). Notebook structure tương tự Phase 3A, chỉ thay model ID và adjust batch size.

### Environment
- **Hardware:** T4 GPU — **BẮT BUỘC**
- **Platform:** Google Colab — **dùng account KHÁC với Phase 3A** (chạy song song để tiết kiệm thời gian)
- **Estimated training time:** ~60-120 phút/epoch ⚠️ > 15 phút, PHẢI có checkpoint
- **Note OOM:** BARTpho-word lớn hơn ViT5, giảm batch_size xuống 2, tăng grad_accum lên 8

### Differences from Phase 3A

```python
# Cell 4 — Different preprocessing for BARTpho
MODEL_NAME     = "vinai/bartpho-word"
MAX_INPUT_LEN  = 1024  # BARTpho supports 1024 tokens
MAX_TARGET_LEN = 256

# BARTpho uses different input format — no "question:" prefix needed
# But we keep consistent format for fair comparison
def preprocess_bartpho(examples):
    inputs = [
        f"question: {q} context: {c}"
        for q, c in zip(examples['question'], examples['context'])
    ]
    targets = examples['answer']
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LEN,
                              padding="max_length", truncation=True)
    labels = tokenizer(text_target=targets, max_length=MAX_TARGET_LEN,
                       padding="max_length", truncation=True)
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

```python
# Cell 6 — OOM-safe config for BARTpho (larger model)
BATCH_SIZE   = 2    # reduced from 4
GRAD_ACCUM   = 8    # effective batch = 2 * 8 = 16 (same as ViT5)
CHECKPOINT_DIR = "/content/bartpho_checkpoints"
BEST_MODEL_DIR = f"{DRIVE_ROOT}/checkpoints/bartpho/best"
```

### Notebook: `03b_train_bartpho.ipynb`

> Copy `03a_train_vit5.ipynb`, đổi MODEL_NAME, batch size, checkpoint paths.  
> Tất cả cell còn lại giữ nguyên logic.

### ✅ Checklist Phase 3B
- [ ] Khác account với Phase 3A (chạy song song)
- [ ] T4 GPU confirmed
- [ ] BARTpho best model saved to `checkpoints/bartpho/best/` on Drive
- [ ] `bartpho_training_log.csv` saved

---

## Phase 3C — Fine-tune mT5-base (Optional/Bonus)

### Description
**OPTIONAL** — Làm nếu còn thời gian và GPU quota. mT5-base là multilingual baseline, kết quả thường thấp hơn Vietnamese-specific models, tạo data point cho ablation study.

### Key Differences
```python
MODEL_NAME     = "google/mt5-base"
MAX_INPUT_LEN  = 512
MAX_TARGET_LEN = 128
BATCH_SIZE     = 2     # 580M params, large
GRAD_ACCUM     = 8
# mT5 uses SentencePiece tokenizer — same API, no code change needed
```

> Chỉ làm Phase 3C nếu Phase 3A và 3B đã hoàn thành và còn > 2 tuần.

---

## Phase 4 — Unified Evaluation & Comparison

### Description
Load tất cả fine-tuned models và baseline, chạy inference trên **test set** (chỉ dùng test set ở đây — không dùng trước đó), tính đầy đủ metrics, export comparison table.

### Environment
- **Hardware:** T4 GPU (để inference nhanh hơn) hoặc CPU nếu dataset nhỏ
- **Platform:** Google Colab
- **Estimated time:** ~20-40 phút

### Input
- `data/processed/test.json`
- `checkpoints/vit5/best/`
- `checkpoints/bartpho/best/`
- `results/baseline_groq.json`

### Output
- `results/eval_vit5.json`
- `results/eval_bartpho.json`
- `results/comparison_table.csv`

### Notebook: `04_evaluation.ipynb`

**Cell 1 — Setup**
```python
from google.colab import drive
drive.mount('/content/drive')
DRIVE_ROOT = "/content/drive/MyDrive/vimedaq-project"

!pip install transformers evaluate rouge-score sacrebleu bert-score sentencepiece -q

import torch, json, pandas as pd, numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
```

**Cell 2 — Load Test Data**
```python
with open(f"{DRIVE_ROOT}/data/processed/test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

questions  = [d['question'] for d in test_data]
contexts   = [d['context']  for d in test_data]
references = [d['answer']   for d in test_data]
topics     = [d['topic']    for d in test_data]
print(f"Test samples: {len(test_data)}")
```

**Cell 3 — Inference Function**
```python
def batch_inference(model, tokenizer, questions, contexts,
                    max_input=512, max_target=128, batch_size=8, device='cuda'):
    """Run batch inference, return list of predicted strings."""
    model.eval()
    model.to(device)
    predictions = []
    for i in range(0, len(questions), batch_size):
        batch_q = questions[i:i+batch_size]
        batch_c = contexts[i:i+batch_size]
        inputs_text = [f"question: {q} context: {c}" for q, c in zip(batch_q, batch_c)]
        inputs = tokenizer(inputs_text, return_tensors="pt",
                           max_length=max_input, padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_target,
                                      num_beams=4, early_stopping=True)
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend([p.strip() for p in preds])
    return predictions
```

**Cell 4 — Evaluate ViT5**
```python
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load ViT5
vit5_tokenizer = AutoTokenizer.from_pretrained(f"{DRIVE_ROOT}/checkpoints/vit5/best")
vit5_model     = AutoModelForSeq2SeqLM.from_pretrained(f"{DRIVE_ROOT}/checkpoints/vit5/best")

vit5_preds = batch_inference(vit5_model, vit5_tokenizer, questions, contexts,
                               max_input=512, max_target=128, device=DEVICE)
print("ViT5 inference done.")

# Free memory
del vit5_model
torch.cuda.empty_cache()
```

**Cell 5 — Evaluate BARTpho**
```python
bartpho_tokenizer = AutoTokenizer.from_pretrained(f"{DRIVE_ROOT}/checkpoints/bartpho/best")
bartpho_model     = AutoModelForSeq2SeqLM.from_pretrained(f"{DRIVE_ROOT}/checkpoints/bartpho/best")

bartpho_preds = batch_inference(bartpho_model, bartpho_tokenizer, questions, contexts,
                                 max_input=1024, max_target=256, device=DEVICE)
print("BARTpho inference done.")
del bartpho_model
torch.cuda.empty_cache()
```

**Cell 6 — Compute All Metrics**
```python
rouge  = evaluate.load("rouge")
bleu   = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

def compute_all_metrics(preds, refs, model_name):
    rouge_r   = rouge.compute(predictions=preds, references=refs)
    bleu_r    = bleu.compute(predictions=preds, references=[[r] for r in refs])
    bs_r      = bertscore.compute(predictions=preds, references=refs, lang="vi")
    return {
        "model":    model_name,
        "rouge1":   round(rouge_r['rouge1'], 4),
        "rouge2":   round(rouge_r['rouge2'], 4),
        "rougeL":   round(rouge_r['rougeL'], 4),
        "bleu4":    round(bleu_r['bleu'], 4),
        "bertscore_f1": round(np.mean(bs_r['f1']), 4),
    }

results = []
results.append(compute_all_metrics(vit5_preds,    references, "ViT5-base (fine-tuned)"))
results.append(compute_all_metrics(bartpho_preds, references, "BARTpho-word (fine-tuned)"))

# Add Groq baseline from saved file
with open(f"{DRIVE_ROOT}/results/baseline_groq.json") as f:
    groq_res = json.load(f)
results.append({
    "model": "Llama-3.3-70B-versatile (zero-shot)",
    "rouge1": groq_res['rouge1'], "rouge2": groq_res['rouge2'],
    "rougeL": groq_res['rougeL'], "bleu4": groq_res['bleu4'],
    "bertscore_f1": "N/A"
})

comparison_df = pd.DataFrame(results)
print(comparison_df.to_string(index=False))
```

**Cell 7 — Per-topic Analysis**
```python
import pandas as pd

# Per-topic ROUGE-L for ViT5 (most granular analysis)
topic_results = []
unique_topics = list(set(topics))

for topic in unique_topics:
    idx = [i for i, t in enumerate(topics) if t == topic]
    t_preds = [vit5_preds[i] for i in idx]
    t_refs  = [references[i]  for i in idx]
    r = rouge.compute(predictions=t_preds, references=t_refs)
    topic_results.append({"topic": topic, "n": len(idx), "rougeL": round(r['rougeL'], 4)})

topic_df = pd.DataFrame(topic_results)
print("\nPer-topic ROUGE-L (ViT5):")
print(topic_df.to_string(index=False))
```

**Cell 8 — Save All Results**
```python
import os
os.makedirs(f"{DRIVE_ROOT}/results", exist_ok=True)

# Comparison table
comparison_df.to_csv(f"{DRIVE_ROOT}/results/comparison_table.csv", index=False)
topic_df.to_csv(f"{DRIVE_ROOT}/results/per_topic_analysis.csv", index=False)

# Detailed predictions for each model
for model_name, preds in [("vit5", vit5_preds), ("bartpho", bartpho_preds)]:
    out = [{"question": questions[i], "reference": references[i],
            "prediction": preds[i], "topic": topics[i]}
           for i in range(len(preds))]
    with open(f"{DRIVE_ROOT}/results/eval_{model_name}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

print("All evaluation results saved to Drive.")
print("\nFinal comparison:")
print(comparison_df.to_string(index=False))
```

### ✅ Checklist Phase 4
- [ ] `comparison_table.csv` saved (ROUGE-1/2/L, BLEU-4, BERTScore-F1)
- [ ] `per_topic_analysis.csv` saved
- [ ] `eval_vit5.json`, `eval_bartpho.json` saved

---

## Phase 5 — Error Analysis & Ablation Study

### Description
Phân tích định tính lỗi mô hình, ablation study nhỏ. Đây là phần phân biệt đồ án điểm 8 vs điểm 10.

### Environment
- **Hardware:** CPU
- **Platform:** Colab hoặc local

### Notebook: `05_error_analysis.ipynb`

**Tasks:**

1. **Best case / Worst case analysis:** 10 mẫu ViT5 predict tốt nhất (ROUGE-L cao) và 10 mẫu tệ nhất — phân tích TẠI SAO

2. **Failure mode categorization:**
   - Factual hallucination (model nói thông tin sai)
   - Length mismatch (answer quá ngắn/quá dài)
   - Topic confusion (trả lời đúng nhưng sai topic)

3. **Ablation: beam search vs greedy decoding**
```python
# Compare num_beams=1 (greedy) vs num_beams=4 (beam search) on val set
```

4. **Learning curve plot:** từ `vit5_training_log.csv`
```python
import pandas as pd, matplotlib.pyplot as plt

log = pd.read_csv(f"{DRIVE_ROOT}/logs/vit5_training_log.csv")
# Plot train loss vs eval rougeL over steps
```

### Output
- `results/error_analysis/` → Drive (charts + CSV)
- Nội dung trực tiếp đưa vào LaTeX report

---

## Phase 6 — HuggingFace Hub Publication

### Description
Push best fine-tuned model lên HuggingFace Hub. **Làm sau khi có final eval results.**

### Environment
- CPU, Colab hoặc local

### Tasks

**Step 1 — Login to HuggingFace**
```python
!pip install huggingface_hub -q
from huggingface_hub import notebook_login
notebook_login()  # Enter your HF token
```

**Step 2 — Push Model**
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

best_model_path = f"{DRIVE_ROOT}/checkpoints/vit5/best"
tokenizer = AutoTokenizer.from_pretrained(best_model_path)
model     = AutoModelForSeq2SeqLM.from_pretrained(best_model_path)

# Replace with your HuggingFace username
HF_REPO = "YOUR_HF_USERNAME/vit5-vimedaq-medical-qa"

tokenizer.push_to_hub(HF_REPO, private=False)
model.push_to_hub(HF_REPO, private=False)
print(f"Model published at: https://huggingface.co/{HF_REPO}")
```

**Step 3 — Write Model Card**

Create `README.md` on HuggingFace with:
- Model description
- Dataset: ViMedAQA (Tran et al., ACL 2024)
- Training details (hyperparams, epochs)
- Evaluation results table (ROUGE/BLEU/BERTScore)
- Usage code snippet
- Citation

---

## Phase 7 — Web Application (Gradio + RAG)

### Description
Gradio app với **RAG pipeline**: user chỉ nhập **câu hỏi** → BM25 tự động truy xuất context → ViT5 sinh câu trả lời.  
UI hiển thị thêm "Retrieved Context" ở output phụ (cho mục đích debug/transparency).  
Deploy trên HuggingFace Spaces (miễn phí, public URL).

### Architecture
```
User Question → BM25 Retrieval (top-1) → "question: {q} context: {c}" → ViT5 → Answer
```

### Prerequisites
- Phase 1.5 hoàn thành: `medical_corpus.json` + `bm25_index.pkl` phải tồn tại
- Phase 6 hoàn thành: model đã push lên HuggingFace Hub

### File: `app/app.py`

```python
"""
ViMedAQA Medical QA Demo
Model: ViT5-base fine-tuned on ViMedAQA dataset (ACL 2024)
"""
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model from HuggingFace Hub
MODEL_ID   = "YOUR_HF_USERNAME/vit5-vimedaq-medical-qa"
MAX_INPUT  = 512
MAX_OUTPUT = 128

print(f"Loading model {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
device    = "cuda" if torch.cuda.is_available() else "cpu"
model     = model.to(device)
model.eval()
print("Model loaded.")


def answer_question(question: str, context: str) -> str:
    """Generate an answer given a question and context."""
    if not question.strip() or not context.strip():
        return "⚠️ Please provide both a question and context."
    
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=MAX_INPUT,
        truncation=True,
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_OUTPUT,
            num_beams=4,
            early_stopping=True,
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# Sample examples from ViMedAQA test set
EXAMPLES = [
    [
        "Triệu chứng của bệnh tiểu đường là gì?",
        "Bệnh tiểu đường là tình trạng cơ thể không sản xuất đủ insulin hoặc không sử dụng insulin hiệu quả. Các triệu chứng phổ biến bao gồm khát nước nhiều, tiểu nhiều, mệt mỏi, và mờ mắt."
    ],
    [
        "Thuốc Paracetamol có tác dụng gì?",
        "Paracetamol là thuốc giảm đau và hạ sốt thông dụng. Thuốc được dùng để điều trị đau nhẹ đến vừa như đau đầu, đau răng, đau cơ, và hạ sốt trong cảm cúm."
    ]
]

demo = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(label="Question (Câu hỏi)", placeholder="Nhập câu hỏi y tế..."),
        gr.Textbox(label="Context (Ngữ cảnh)", lines=5,
                   placeholder="Nhập đoạn văn bản y tế liên quan..."),
    ],
    outputs=gr.Textbox(label="Answer (Câu trả lời)"),
    title="🏥 Vietnamese Medical QA",
    description=(
        "**ViT5-base fine-tuned on ViMedAQA** — Vietnamese Medical Abstractive QA dataset "
        "(Tran et al., ACL 2024). Enter a medical question and relevant context to get an answer."
    ),
    examples=EXAMPLES,
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(share=False)
```

### Deploy to HuggingFace Spaces

1. Tạo Space mới tại https://huggingface.co/spaces (Gradio template)
2. Upload `app.py` và `requirements.txt`:
```
gradio==4.31.0
transformers==4.40.0
torch==2.2.0
sentencepiece==0.2.0
```
3. Set `MODEL_ID` đúng với repo đã push ở Phase 6
4. Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/vimedaq-medical-qa`

### ✅ Checklist Phase 7
- [ ] `app.py` chạy được local
- [ ] Deployed on HuggingFace Spaces
- [ ] Public URL hoạt động
- [ ] Examples hiển thị đúng

---

## Phase 8 — LaTeX Report

### Description
Viết báo cáo học thuật bằng LaTeX. Toàn bộ số liệu lấy trực tiếp từ `results/` folder trên Drive.

### Structure (`report/main.tex`)

```
1. Introduction            (~0.5 trang)
   - Motivation: Vietnamese medical NLP gap
   - Contribution: fine-tuning Transformer models on ViMedAQA

2. Related Work            (~0.5 trang)
   - Transformer architecture (Vaswani et al., 2017)
   - T5 (Raffel et al., 2020), BART (Lewis et al., 2020)
   - ViT5 (Phan et al., 2022), BARTpho (Tran et al., 2021)
   - ViMedAQA (Tran et al., 2024)

3. Dataset                 (~1 trang)
   - ViMedAQA description: 4 topics, size, splits
   - Statistics table: length distributions, topic distribution
   - Include EDA figures from data/eda/

4. Methodology             (~1.5 trang)
   - Problem formulation: Abstractive QA definition
   - Model architectures (with Transformer diagram reference)
   - Fine-tuning setup: hyperparameters, training procedure
   - Evaluation metrics: ROUGE-1/2/L, BLEU-4, BERTScore

5. Experiments & Results   (~1.5 trang)
   - Comparison table (all models)
   - Per-topic analysis table
   - Learning curves (ViT5 and BARTpho)
   - Error analysis examples (3-5 cases)

6. Ablation Study          (~0.5 trang)
   - Beam search vs greedy
   - Effect of max_length

7. Conclusion              (~0.3 trang)
   - Summary of findings
   - Limitations & future work (RAG extension)
```

### Key References (bibtex)

```bibtex
@inproceedings{tran-etal-2024-vimedaqa,
  title     = "{ViMedAQA}: A Vietnamese Medical Abstractive Question-Answering Dataset",
  author    = "Tran, Minh-Nam and Nguyen, Phu-Vinh and Nguyen, Long and Dinh, Dien",
  booktitle = "Proceedings of the 62nd Annual Meeting of ACL (Student Research Workshop)",
  year      = "2024",
  pages     = "252--260",
}

@inproceedings{phan-etal-2022-vit5,
  title     = "{V}i{T}5: Pretrained Text-to-Text Transformer for {V}ietnamese Language Generation",
  author    = "Phan, Long and Tran, Hieu and Nguyen, Hieu and Trinh, Trieu H.",
  booktitle = "Proceedings of NAACL 2022 (Student Research Workshop)",
  year      = "2022",
  pages     = "136--142",
}

@inproceedings{tran-etal-2021-bartpho,
  title     = "{BART}pho: Pre-trained Sequence-to-Sequence Models for {V}ietnamese",
  author    = "Tran, Nguyen Luong and Le, Duong Minh and Nguyen, Dat Quoc",
  booktitle = "Proceedings of EMNLP 2021 (Findings)",
  year      = "2021",
}

@inproceedings{vaswani-etal-2017-attention,
  title  = "Attention is All You Need",
  author = "Vaswani, Ashish and others",
  year   = "2017",
  booktitle = "Advances in Neural Information Processing Systems",
}
```

---

## Hardware & Account Strategy

| Phase | Hardware | Platform | Account |
|---|---|---|---|
| 0 — Setup | CPU | Local | Any |
| 1 — EDA | CPU | Colab Free | Account 1 |
| 2 — Baseline | CPU | Colab Free | Account 1 |
| 3A — ViT5 | **T4 GPU** | Colab Free | Account 2 |
| 3B — BARTpho | **T4 GPU** | Colab Free | Account 3 |
| 3C — mT5 (opt) | **T4 GPU** | Colab Free | Account 4 |
| 4 — Evaluation | T4 GPU | Colab Free | Account 2 or 3 |
| 5 — Error Analysis | CPU | Colab Free | Account 1 |
| 6 — HF Hub | CPU | Colab/Local | Any |
| 7 — Web App | CPU | HF Spaces | Any |
| 8 — Report | CPU | Local (LaTeX) | Any |

### Colab GPU Quota Management
- Colab Free: ~38-40 GPU hours/month per account
- Phase 3A (ViT5, 5 epochs): ~5-7 GPU hours
- Phase 3B (BARTpho, 5 epochs): ~7-10 GPU hours
- Phase 4 (Evaluation): ~1-2 GPU hours
- **Total needed per model: ~6-12 hours** → 1 account per model is sufficient

---

## Git & Drive Strategy

### Git Commit Convention
```
feat: add ViT5 fine-tuning notebook
fix: resolve OOM issue in BARTpho preprocessing
docs: update README with evaluation results
data: add dataset EDA statistics
model: push ViT5-base fine-tuned to HuggingFace Hub
```

### What NOT to commit to GitHub
```gitignore
# .gitignore
*.pt
*.bin
*.safetensors
data/
*.csv
*.json
__pycache__/
.env
wandb/
```

### Redundancy Strategy
- **Google Drive** = primary storage for checkpoints, data, results
- **GitHub** = code, notebooks (no data, no weights)
- **HuggingFace Hub** = published model weights (public)

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Colab disconnects during training | High | High | Checkpoint every 100 steps → Drive |
| OOM on T4 with BARTpho | Medium | High | Reduce batch_size=2, grad_accum=8, fp16=True |
| ViMedAQA has no split → need manual split | Medium | Low | Stratified split by topic, seed=42 |
| Groq API Tokens Per Day (TPD) Limit | Medium | Low | Sử dụng cơ chế Checkpointing và thiết lập luân phiên nhiều API Key |
| BERTScore slow on CPU | Low | Low | Use GPU or compute offline |
| HF Spaces RAM limit (16GB free) | Low | Medium | Load model to CPU, use quantization if needed |

---

## Timeline (8 Weeks)

| Week | Milestone |
|---|---|
| 1 | Phase 0 + Phase 1: Setup & EDA complete |
| 2 | Phase 2: Groq baseline complete |
| 3-4 | Phase 3A + 3B: Both models training (parallel) |
| 5 | Phase 4: Full evaluation & comparison table |
| 5-6 | Phase 5: Error analysis & ablation |
| 6 | Phase 6: HuggingFace Hub publication |
| 6-7 | Phase 7: Web app & deployment |
| 7-8 | Phase 8: LaTeX report writing |

---

*Pipeline designed for ViMedAQA (ACL 2024) × HCMUS Statistical Learning course. All code in English. Cite ViMedAQA, ViT5, BARTpho authors appropriately in report.*
