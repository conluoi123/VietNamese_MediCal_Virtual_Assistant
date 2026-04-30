"""
ViMedAQA Medical QA Demo — RAG Mode
Model: ViT5-base fine-tuned on ViMedAQA dataset (ACL 2024)
Retrieval: BM25Okapi over medical context corpus

Architecture:
  User Question → BM25 Retrieval (top-1 context) → ViT5 Generation → Answer

The user only needs to input a question. The system automatically retrieves
the most relevant medical context from the knowledge base.
"""

import os
import json
import pickle

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rank_bm25 import BM25Okapi


# ===========================================================================
# Configuration
# ===========================================================================
MODEL_ID    = "YOUR_HF_USERNAME/vit5-vimedaq-medical-qa"  # Update after Phase 6
MAX_INPUT   = 512
MAX_OUTPUT  = 128

# Paths to retrieval index files (created in Phase 1.5)
# Adjust these paths based on your deployment environment:
#   - Local dev:  relative to this file (app/ directory)
#   - HF Spaces:  bundle these files alongside app.py
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "medical_corpus.json")
BM25_PATH   = os.path.join(BASE_DIR, "..", "data", "processed", "bm25_index.pkl")


# ===========================================================================
# Load Retrieval Index (once at startup)
# ===========================================================================
print("Loading retrieval index...")

if not os.path.exists(CORPUS_PATH):
    raise FileNotFoundError(
        f"⚠️ PREREQUISITE FAILED: '{CORPUS_PATH}' not found.\n"
        f"Run Phase 1.5 (01b_build_retrieval_index.ipynb) first to build the knowledge base."
    )
if not os.path.exists(BM25_PATH):
    raise FileNotFoundError(
        f"⚠️ PREREQUISITE FAILED: '{BM25_PATH}' not found.\n"
        f"Run Phase 1.5 (01b_build_retrieval_index.ipynb) first to build the BM25 index."
    )

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)

with open(BM25_PATH, "rb") as f:
    bm25_index = pickle.load(f)

print(f"✅ Retrieval index loaded: {len(corpus)} medical documents")


# ===========================================================================
# Load QA Model (once at startup)
# ===========================================================================
print(f"Loading model {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
device    = "cuda" if torch.cuda.is_available() else "cpu"
model     = model.to(device)
model.eval()
print(f"✅ Model loaded on {device}")


# ===========================================================================
# Core Functions
# ===========================================================================
def retrieve_context(question: str) -> str:
    """Retrieve the most relevant medical context for a given question using BM25."""
    tokenized_query = question.split()
    scores = bm25_index.get_scores(tokenized_query)
    top_idx = scores.argsort()[-1]  # Top-1 best match
    return corpus[top_idx]


def answer_question(question: str) -> tuple[str, str]:
    """
    End-to-end RAG pipeline:
    1. Retrieve context via BM25
    2. Format input as training format: "question: {q} context: {c}"
    3. Generate answer with ViT5

    Returns:
        tuple: (generated_answer, retrieved_context)
    """
    if not question.strip():
        return "⚠️ Please enter a medical question.", ""

    # Step 1: Retrieve context
    retrieved_context = retrieve_context(question)

    # Step 2: Format input (must match training format exactly)
    input_text = f"question: {question} context: {retrieved_context}"

    # Step 3: Generate answer
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

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer, retrieved_context


# ===========================================================================
# Gradio Interface
# ===========================================================================
EXAMPLES = [
    ["Triệu chứng của bệnh tiểu đường là gì?"],
    ["Thuốc Paracetamol có tác dụng gì?"],
    ["Gan có chức năng gì trong cơ thể?"],
    ["Bệnh viêm phổi có nguy hiểm không?"],
]

demo = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(
            label="Question (Câu hỏi y tế)",
            placeholder="Nhập câu hỏi y tế bằng tiếng Việt...",
            lines=2,
        ),
    ],
    outputs=[
        gr.Textbox(label="Answer (Câu trả lời)", lines=3),
        gr.Textbox(label="📄 Retrieved Context (Debug — Ngữ cảnh được truy xuất)", lines=5),
    ],
    title="🏥 Vietnamese Medical QA — RAG Pipeline",
    description=(
        "**ViT5-base fine-tuned on ViMedAQA** (Tran et al., ACL 2024).\n\n"
        "Just enter a medical question — the system automatically retrieves relevant "
        "medical context from the knowledge base using BM25, then generates an answer.\n\n"
        "The 'Retrieved Context' field shows which document was used (for transparency/debugging)."
    ),
    examples=EXAMPLES,
    theme=gr.themes.Soft(),
    allow_flagging="never",
)


if __name__ == "__main__":
    demo.launch(share=False)
