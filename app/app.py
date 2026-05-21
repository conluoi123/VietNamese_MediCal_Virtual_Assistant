import os
import json
import gc
import numpy as np

import gradio as gr
import torch
# Disable autograd engine globally for inference to save memory
torch.set_grad_enabled(False)

import sentencepiece as spm
from transformers import AutoModelForSeq2SeqLM
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
import faiss


# ===========================================================================
# Configuration
# ===========================================================================
MODEL_ID    = "ntthanh0307/vit5-vimedaq-medical-qa"
MAX_INPUT   = 512
MAX_OUTPUT  = 128

# Paths to retrieval index files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Đường dẫn Local
local_corpus = os.path.join(BASE_DIR, "..", "data", "processed", "medical_corpus.json")
local_faiss  = os.path.join(BASE_DIR, "..", "data", "processed", "faiss_index.bin")

# 2. Đường dẫn Google Drive (nếu đang chạy trên Colab)
drive_corpus = "/content/drive/MyDrive/vimedaq-project/data/processed/medical_corpus.json"
drive_faiss  = "/content/drive/MyDrive/vimedaq-project/data/processed/faiss_index.bin"

if os.path.exists(local_corpus) and os.path.exists(local_faiss):
    CORPUS_PATH = local_corpus
    FAISS_PATH  = local_faiss
elif os.path.exists(drive_corpus) and os.path.exists(drive_faiss):
    CORPUS_PATH = drive_corpus
    FAISS_PATH  = drive_faiss
else:
    # 3. Đường dẫn dự phòng cho HuggingFace Spaces
    CORPUS_PATH = os.path.join(BASE_DIR, "medical_corpus.json")
    FAISS_PATH  = os.path.join(BASE_DIR, "faiss_index.bin")


# ===========================================================================
# Load Retrieval Index (once at startup)
# ===========================================================================
print("Loading retrieval index...")

if not os.path.exists(CORPUS_PATH):
    raise FileNotFoundError(
        f"⚠️ PREREQUISITE FAILED: '{CORPUS_PATH}' not found.\n"
        f"Run Phase 1.6 to build the knowledge base."
    )
if not os.path.exists(FAISS_PATH):
    raise FileNotFoundError(
        f"⚠️ PREREQUISITE FAILED: '{FAISS_PATH}' not found.\n"
        f"Run Phase 1.6 to build the FAISS index."
    )

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus = json.load(f)

faiss_index = faiss.read_index(FAISS_PATH)

print(f"✅ Retrieval index loaded: {len(corpus)} medical documents")


# ===========================================================================
# Load SentencePiece tokenizer from VietAI/vit5-base (correct vocab)
# ===========================================================================
print("Downloading SentencePiece model from VietAI/vit5-base...")
spiece_path = hf_hub_download(repo_id="VietAI/vit5-base", filename="spiece.model")
sp = spm.SentencePieceProcessor()
sp.Load(spiece_path)

# T5 special token IDs (match training setup)
PAD_ID = 0
EOS_ID = 1
UNK_ID = 2

print(f"✅ SentencePiece tokenizer loaded: vocab_size={sp.GetPieceSize()}")


# ===========================================================================
# Load QA Model (once at startup)
# ===========================================================================
print(f"Loading model {MODEL_ID}...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
model.config.decoder_start_token_id = PAD_ID
if model.generation_config is not None:
    model.generation_config.decoder_start_token_id = PAD_ID
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
print(f"✅ Model loaded on {device}")

print(f"Loading retriever model 'bkai-foundation-models/vietnamese-bi-encoder'...")
retriever_model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder", device=device)
print(f"✅ Retriever model loaded")


# ===========================================================================
# Helper: SentencePiece encode/decode (matching T5Tokenizer behavior)
# ===========================================================================
def sp_encode(text, max_length=512):
    """Encode text using SentencePiece, add EOS, pad/truncate to max_length."""
    ids = sp.EncodeAsIds(text)
    # Truncate (leave room for EOS)
    if len(ids) > max_length - 1:
        ids = ids[:max_length - 1]
    ids.append(EOS_ID)  # T5 adds </s> at end
    return ids


def sp_decode(ids):
    """Decode token IDs back to text, skipping special tokens."""
    # Filter out special tokens (pad=0, eos=1, unk=2)
    clean_ids = [i for i in ids if i not in (PAD_ID, EOS_ID)]
    return sp.DecodeIds(clean_ids)


# ===========================================================================
# Core Functions
# ===========================================================================
TOP_K = 3  # Number of contexts to retrieve


def retrieve_contexts(question: str, top_k: int = TOP_K) -> list[str]:
    """Retrieve top-k contexts using Vietnamese Bi-Encoder + FAISS."""
    # 1. Encode query and normalize vector (since FAISS IndexFlatIP expects L2 normalized vectors for Cosine Similarity)
    query_emb = retriever_model.encode([question], normalize_embeddings=True, show_progress_bar=False)
    query_emb_f32 = np.array(query_emb).astype('float32')
    
    # 2. Search FAISS index
    distances, indices = faiss_index.search(query_emb_f32, top_k)
    
    # 3. Get corresponding texts from corpus
    matched_contexts = []
    for idx in indices[0]:
        if 0 <= idx < len(corpus):
            matched_contexts.append(corpus[idx])
            
    return matched_contexts


def answer_question(question: str, num_beams: int = 1, max_new_tokens: int = 128, top_k: int = 3) -> tuple[str, str]:
    """
    End-to-end RAG pipeline:
    1. Retrieve top-k contexts via BM25 / dense retriever
    2. Combine contexts and generate answer with ViT5

    Returns:
        tuple: (generated_answer, retrieved_contexts_display)
    """
    # 0. Clean memory before execution to avoid compounding memory pressure
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not question.strip():
        return "⚠️ Please enter a medical question.", ""

    # Step 1: Retrieve top-k contexts
    contexts = retrieve_contexts(question, top_k=top_k)
    contexts_display = "\n\n".join(
        [f"--- Context {i+1} ---\n{ctx}" for i, ctx in enumerate(contexts)]
    )

    # Step 2: Combine contexts to give model maximum flexibility
    combined_context = " ".join(contexts)
    input_text = f"question: {question} context: {combined_context}"

    # Step 3: Encode and Generate
    input_ids = sp_encode(input_text, max_length=MAX_INPUT)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask = torch.ones_like(input_tensor).to(device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True if num_beams > 1 else False,
                decoder_start_token_id=PAD_ID,
                forced_bos_token_id=None,
                use_cache=True,
            )

        output_ids = outputs[0].tolist()
        answer = sp_decode(output_ids).strip()
        
        if not answer:
            answer = "Xin lỗi, tôi không tìm được câu trả lời phù hợp dựa trên ngữ cảnh hiện tại."
    finally:
        # Step 4: Ensure all torch tensors are deleted and GC/cache clearing is triggered
        try:
            del input_tensor
            del attention_mask
            del outputs
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[DEBUG] Generated answer: '{answer}'")
    return answer, contexts_display


# ===========================================================================
# Gradio Interface
# ===========================================================================
EXAMPLES = [
    ["Triệu chứng của bệnh tiểu đường là gì?"],
    ["Thuốc Paracetamol có tác dụng gì?"],
    ["Gan có chức năng gì trong cơ thể?"],
    ["Bệnh viêm phổi có nguy hiểm không?"],
]

custom_css = """
.header-container {
    text-align: center;
    margin-bottom: 25px;
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.08), rgba(16, 185, 129, 0.08));
    padding: 35px 20px;
    border-radius: 16px;
    border: 1px solid rgba(6, 182, 212, 0.15);
}
.header-title {
    font-family: 'Outfit', 'Inter', sans-serif !important;
    background: linear-gradient(135deg, #0ea5e9, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    font-size: 2.4rem !important;
    margin-bottom: 12px !important;
    letter-spacing: -0.5px;
}
.header-subtitle {
    color: #4b5563;
    font-size: 1.15rem;
    max-width: 800px;
    margin: 0 auto;
    line-height: 1.6;
}
.submit-btn {
    background: linear-gradient(135deg, #0ea5e9, #10b981) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 6px -1px rgba(6, 182, 212, 0.15), 0 2px 4px -1px rgba(6, 182, 212, 0.06) !important;
}
.submit-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 15px -3px rgba(6, 182, 212, 0.25), 0 4px 6px -2px rgba(6, 182, 212, 0.15) !important;
    filter: brightness(1.05);
}
.accordion-settings {
    border: 1px solid rgba(209, 213, 219, 0.6) !important;
    border-radius: 12px !important;
    background-color: rgba(249, 250, 251, 0.5) !important;
    margin-top: 15px !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("""
        <div class="header-container">
            <h1 class="header-title">🏥 ViMedAQA — Trợ lý Ảo Y tế Tiếng Việt</h1>
            <p class="header-subtitle">
                Hệ thống hỏi đáp y khoa tự động kết hợp tìm kiếm ngữ cảnh ngữ nghĩa (Dense RAG) 
                và sinh câu trả lời bằng mô hình <strong>ViT5-base</strong> (ACL 2024).
            </p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="Câu hỏi y tế (Question)",
                placeholder="Nhập câu hỏi y tế bằng tiếng Việt (ví dụ: Triệu chứng bệnh tiểu đường là gì?)...",
                lines=3
            )
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Xóa", variant="secondary")
                submit_btn = gr.Button("⚡ Gửi câu hỏi", variant="primary", elem_classes="submit-btn")
            
            gr.Examples(
                examples=EXAMPLES,
                inputs=[question_input],
                label="💡 Câu hỏi ví dụ gợi ý"
            )
            
        with gr.Column(scale=2):
            with gr.Accordion("⚙️ Cấu hình sinh & Tối ưu RAM/CPU", open=True, elem_classes="accordion-settings"):
                num_beams_slider = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=1,
                    step=1,
                    label="Tối ưu Beam Search (num_beams)",
                    info="1 = Greedy Search (Khuyên dùng: Cực nhanh, tiết kiệm RAM/CPU). 4 = Beam Search (Chất lượng cao, tốn tài nguyên)."
                )
                max_tokens_slider = gr.Slider(
                    minimum=32,
                    maximum=256,
                    value=128,
                    step=8,
                    label="Độ dài câu trả lời tối đa (Max new tokens)"
                )
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Số lượng văn bản truy xuất (Top-K Contexts)"
                )
                
    with gr.Row():
        with gr.Column(scale=1):
            answer_output = gr.Textbox(
                label="🏥 Câu trả lời từ Trợ lý Ảo (Generated Answer)",
                lines=5,
                interactive=False
            )
            
        with gr.Column(scale=1):
            context_output = gr.Textbox(
                label="📄 Ngữ cảnh y tế tham khảo (Retrieved Contexts)",
                lines=8,
                interactive=False
            )

    # Set actions
    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, num_beams_slider, max_tokens_slider, top_k_slider],
        outputs=[answer_output, context_output]
    )
    question_input.submit(
        fn=answer_question,
        inputs=[question_input, num_beams_slider, max_tokens_slider, top_k_slider],
        outputs=[answer_output, context_output]
    )
    
    # Lambda function to clear textboxes
    clear_btn.click(
        fn=lambda: ("", "", ""),
        inputs=[],
        outputs=[question_input, answer_output, context_output]
    )


if __name__ == "__main__":
    demo.launch(share=False)
