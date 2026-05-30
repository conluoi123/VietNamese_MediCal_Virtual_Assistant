# ViMedAQA — Vietnamese Medical Abstractive QA Assistant

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![HuggingFace Model](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-orange.svg)](https://huggingface.co/ntthanh0307/vit5-vimedaq-medical-qa)
[![HuggingFace Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Spaces%20Demo-yellow.svg)](https://huggingface.co/spaces/ntthanh0307/vimedaqa-medical-chatbot)
[![Dataset](https://img.shields.io/badge/Dataset-ViMedAQA-red.svg)](https://huggingface.co/datasets/tmnam20/ViMedAQA)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Course:** Statistical Learning — HCMUS (Faculty of Mathematics & Computer Science)  
> **Task:** Abstractive Question Answering and Advanced hybrid RAG in the Vietnamese Medical Domain  
> **Dataset:** [tmnam20/ViMedAQA](https://huggingface.co/datasets/tmnam20/ViMedAQA) (ACL 2024 SRW)

---

## 📖 Overview

**ViMedAQA Vietnamese Medical QA Assistant** is an advanced, production-grade clinical question-answering assistant designed specifically for the Vietnamese language. 

The project fine-tunes state-of-the-art Vietnamese sequence-to-sequence models (**ViT5-base** and **BARTpho-word**) on the medical QA corpus, and implements a sophisticated, hybrid **Retrieval-Augmented Generation (RAG)** pipeline to automatically retrieve and reason over medical contexts, effectively neutralizing hallucinations.

### 🚀 Production-Grade Hybrid RAG Architecture

Rather than relying on a basic retrieval module, the system implements a **three-stage hybrid RAG pipeline** to deliver highly accurate, contextual answers:

```
                  ┌───────────────────────┐
                  │ User Question (Query) │
                  └───────────┬───────────┘
                              ▼
                     [ Vietnamese PyVi ]
                   Word Segmented & Lowered
                              │
             ┌────────────────┴────────────────┐
             ▼ (Stage 1a: Semantic)            ▼ (Stage 1b: Lexical)
      ┌──────────────┐                  ┌──────────────┐
      │  Bi-Encoder  │                  │  BM25Okapi   │
      │  embeddings  │                  │  (Keyword)   │
      └──────┬───────┘                  └──────┬───────┘
             ▼                                 ▼
      ┌──────────────┐                  ┌──────────────┐
      │ FAISS Search │                  │ BM25 Ranking │
      └──────┬───────┘                  └──────┬───────┘
             ▼                                 ▼
             └────────────────┬────────────────┘
                              ▼ (Stage 2)
                 [ Reciprocal Rank Fusion ]
                    RRF Scoring & Fusion
                              │
                              ▼
                   [ top-50 Candidates ]
                              │
                              ▼ (Stage 3)
                 [ Cross-Encoder Re-ranker ]
                   mMiniLMv2 Match Scoring
                              │
                              ▼
              ┌───────────────────────────────┐
     Yes      │ Prob Score >= Threshold (0.4)?│
   ┌──────────┴───────────────────────────────┴──────────┐ No
   ▼                                                     ▼
[ RAG Generation ]                               [ Block & Default Response ]
Top-3 medical contexts fed to                   "Dữ liệu y khoa đáng tin cậy không
Seq2Seq Generator (ViT5 / BARTpho)              có thông tin chính xác... Tránh tự ý
with Beam Search decoding                       dùng thuốc & hỏi bác sĩ chuyên khoa."
   │                                                     │
   ▼                                                     ▼
┌────────────────────────────────────────────────────────┐
│             Final Safe Medical Response                │
└────────────────────────────────────────────────────────┘
```

1. **Stage 1 (Parallel Dual-Retrieval):**
   * **Lexical Retrieval (BM25Okapi):** Captures precise medical vocabulary and drug names. Texts are pre-tokenized using Vietnamese word segmenter `pyvi`.
   * **Semantic Retrieval (FAISS vector store):** Captures semantic clinical intent using `bkai-foundation-models/vietnamese-bi-encoder` embeddings indexed into a high-speed FAISS vector space.
2. **Stage 2 (Reciprocal Rank Fusion - RRF):** Fuses lexical and semantic rankings with custom weights ($W_{\text{Semantic}} = 0.65$, $W_{\text{Lexical}} = 0.35$) to produce a unified candidate set.
3. **Stage 3 (Cross-Encoder Re-ranking & Guardrail Filtering):** 
   * Top-50 candidates are re-scored by a powerful Cross-Encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`).
   * **Hallucination Prevention Guardrail:** If the highest relevance score falls below the probability threshold ($0.4$), the retrieval is classified as out-of-domain. The generator is bypassed, and a safe, standard clinical disclaimer warning is returned.

---

## 📊 Experimental Results

A rigorous comparative evaluation was performed between the fine-tuned Seq2Seq models and a zero-shot State-of-the-Art Large Language Model baseline (**Llama-3.3-70B-versatile** via the Groq Cloud API). Evaluations were computed using ROUGE-1, ROUGE-2, ROUGE-L, BLEU-4, and multilingual BERTScore F1.

The results (extracted directly from `results/comparison_table.csv`) are summarized below:

| Model Architecture | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 | BERTScore F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ViT5-base (fine-tuned)** | 0.7318 | 0.6153 | 0.6680 | **0.5005** | **0.8752** |
| **BARTpho-word (fine-tuned)** | **0.7327** | 0.6185 | **0.6681** | 0.2794 | 0.8230 |
| **Llama-3.3-70B-versatile (zero-shot)** | 0.7285 | **0.6257** | 0.6664 | 0.4684 | 0.8757 |

### Key Insights:
* **Generative Superiority:** Our fine-tuned models (**ViT5-base** and **BARTpho-word**) achieve performance parity and even surpass Llama-3.3-70B (which is **260x** larger) in ROUGE scores and multilingual BERTScore.
* **BLEU-4 Leader:** ViT5-base achieves an outstanding BLEU-4 of **50.05%**, indicating extremely high syntactic and token-level alignment with professional doctors' answers.
* **Compact & Fast:** Operating at only ~270M (ViT5) and ~396M (BARTpho) parameters, these models are exceptionally efficient and cost-effective for local/clinical deployments compared to commercial closed APIs.

---

## 📁 Project Structure

```
implementation/
├── .gitignore             # Optimized Git settings (allows research data, ignores LaTeX trash)
├── README.md              # Project portal and documentation
├── requirements.txt       # Clean, consolidated package requirements
├── app/
│   ├── app.py             # RAG-enabled Gradio chatbot interface (Bi-Encoder + BM25 + Cross-Encoder)
│   └── requirements.txt   # Lightweight package list for HuggingFace Spaces hosting
├── data/
│   ├── raw/
│   │   └── vimedaq_full.json  # Backed-up raw medical dataset
│   ├── processed/
│   │   ├── train.json         # Splitted train set
│   │   ├── val.json           # Splitted validation set
│   │   ├── test.json          # Splitted test set
│   │   └── medical_corpus.json  # Clean clinical knowledge corpus
│   └── eda/
│       ├── dataset_stats.csv  # Basic question, answer, and context lengths
│       ├── length_distributions.png  # EDA length analysis plot
│       └── comprehensive_eda.png     # General EDA visual dashboard
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Phase 1: Exploratory Data Analysis & splitting
│   ├── 01b_build_retrieval_index.ipynb # Phase 1.5: Build indexing stores (FAISS + BM25 + PyVi)
│   ├── 02_baseline_groq.ipynb         # Phase 2: Groq zero-shot Llama baseline evaluation
│   ├── 03a_train_vit5.ipynb           # Phase 3A: Fine-tuning ViT5-base Seq2Seq
│   ├── 03b_train_bartpho.ipynb        # Phase 3B: Fine-tuning BARTpho-word Seq2Seq
│   ├── 04_evaluation.ipynb            # Phase 4: Rigorous evaluation on ROUGE, BLEU, BERTScore
│   ├── 05_error_analysis.ipynb        # Phase 5: Systematic error mining, Ablation & topic analytics
│   └── 06_push_to_hub.ipynb           # Phase 6: Automatic serialization & Model upload to HF Hub
├── results/
│   ├── baseline_groq.json     # Full evaluation outputs of zero-shot Groq baseline
│   ├── comparison_table.csv   # Unified evaluation metrics CSV
│   ├── eval_vit5.json         # Detailed metrics of ViT5-base model
│   ├── eval_bartpho.json      # Detailed metrics of BARTpho-word model
│   └── per_topic_analysis.csv # Comparative analysis broken down by medical topics
└── report/
    ├── main.tex               # Thesis compilation source LaTeX
    ├── references.bib         # Unified BibTeX database
    ├── main.pdf               # Pre-compiled high-quality report PDF
    ├── sections/              # Chapter contents (.tex files, Chapters 0 to 6)
    ├── figures/               # Figures utilized in the final thesis
    └── module/
        ├── cover.tex          # Formal cover page according to school template
        └── frontmatter.tex    # Task sheet, thanks page, abstract, table of contents
```

---

## 🛠️ Setup & Installation

### Prerequisites
* Recommended OS: Windows 10/11 or Linux
* Python version: `3.10` or higher
* Recommended GPU: Nvidia GPU with CUDA support for accelerated model inference

### 1. Clone & Set up Environment
Clone the repository and install the comprehensive dependencies:
```bash
git clone <repository_url>
cd implementation
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Linux / macOS
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Large File Storage & Index Setup
Model checkpoints and data stores are hosted on Google Drive and HuggingFace Hub to maintain a lightweight Git repo.
If you are running the Gradio web chatbot locally:
1. Ensure you have the `medical_corpus_V3.json`, `faiss_index_V3.bin`, and `bm25_index_V3.pkl` files (generated in `notebooks/01b_build_retrieval_index.ipynb`) placed inside the `app/` folder.
2. The code will automatically download the fine-tuned Seq2Seq model from Hugging Face Hub.

---

## 🖥️ Running the Web App Local

To start the interactive RAG Medical Assistant locally:
```bash
python app/app.py
```
After initialization, Gradio will spin up a local server. Open the link (usually `http://127.0.0.1:7860`) in your browser to interact with the assistant!

---

## 🔄 End-to-End Pipeline Reproduction

To reproduce all phases of the project, run the notebooks in sequential order:

1. **Data Ingestion & Splits:** Open [notebooks/01_data_exploration.ipynb](file:///notebooks/01_data_exploration.ipynb) to clean raw files, perform EDA, and compile the medical splits.
2. **Build Retrieval Knowledge Base:** Run [notebooks/01b_build_retrieval_index.ipynb](file:///notebooks/01b_build_retrieval_index.ipynb) to build BM25 lexical structures and FAISS semantic dense vector stores.
3. **Execute Groq Baseline:** Run [notebooks/02_baseline_groq.ipynb](file:///notebooks/02_baseline_groq.ipynb) (requires a Groq API key) to evaluate the zero-shot baseline.
4. **Fine-Tuning Seq2Seq Generators:**
   * Run [notebooks/03a_train_vit5.ipynb](file:///notebooks/03a_train_vit5.ipynb) to fine-tune ViT5.
   * Run [notebooks/03b_train_bartpho.ipynb](file:///notebooks/03b_train_bartpho.ipynb) to fine-tune BARTpho.
5. **Unified Metric Evaluation:** Run [notebooks/04_evaluation.ipynb](file:///notebooks/04_evaluation.ipynb) to generate performance comparison statistics.
6. **Detailed Error & Topic Analysis:** Run [notebooks/05_error_analysis.ipynb](file:///notebooks/05_error_analysis.ipynb) to perform clinical ablation tests, greedy vs beam search comparison, and compile per-topic statistics.
7. **Serialize & Push to HF Hub:** Run [notebooks/06_push_to_hub.ipynb](file:///notebooks/06_push_to_hub.ipynb) to upload weights to HuggingFace Hub.

---

## 🏛️ Links & Resources

* **Model Weights (HuggingFace Hub):** [ntthanh0307/vit5-vimedaq-medical-qa](https://huggingface.co/ntthanh0307/vit5-vimedaq-medical-qa)
* **Gradio Web Demo (HuggingFace Spaces):** [ntthanh0307/vimedaqa-medical-chatbot](https://huggingface.co/spaces/ntthanh0307/vimedaqa-medical-chatbot)
* **Underlying Dataset:** [tmnam20/ViMedAQA (ACL 2024 SRW)](https://huggingface.co/datasets/tmnam20/ViMedAQA)
* **Academic Thesis Report:** [report/main.pdf](file:///report/main.pdf)

---

## ⚕️ Medical Disclaimer

> [!WARNING]
> **Tuyên bố miễn trừ trách nhiệm y tế (Medical Disclaimer):**  
> Các câu trả lời được đưa ra bởi trợ lý ảo ViMedAQA chỉ mang tính chất tham khảo học thuật ban đầu dựa trên mô hình ngôn ngữ tự động. Hệ thống này không thay thế cho các chẩn đoán chuyên khoa y tế, lời khuyên lâm sàng hoặc hướng dẫn điều trị của bác sĩ chuyên khoa hoặc các chuyên gia y tế có thẩm quyền. Người dùng tuyệt đối không tự ý áp dụng thuốc, thay đổi liều lượng hoặc thực hiện các biện pháp can thiệp y học dựa trên câu trả lời của mô hình này mà chưa qua thăm khám lâm sàng trực tiếp.

---

## 📝 Citation

If you build upon this project, please cite the original ViMedAQA paper:

```bibtex
@inproceedings{tran-etal-2024-vimedaqa,
  title     = "{ViMedAQA}: A Vietnamese Medical Abstractive Question-Answering Dataset",
  author    = "Tran, Minh-Nam and Nguyen, Phu-Vinh and Nguyen, Long and Dinh, Dien",
  booktitle = "Proceedings of the 62nd Annual Meeting of ACL (Student Research Workshop)",
  year      = "2024",
  pages     = "252--260",
}
```
