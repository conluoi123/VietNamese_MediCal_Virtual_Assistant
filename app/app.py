import os
import json
import gc
import math
import numpy as np
import pickle

import gradio as gr
import torch
# Tắt autograd toàn cục để tiết kiệm bộ nhớ khi inference
torch.set_grad_enabled(False)

import sentencepiece as spm
from transformers import AutoModelForSeq2SeqLM
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from pyvi import ViTokenizer

# ===========================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN & THAM SỐ
# ===========================================================================
MODEL_ID    = "ntthanh0307/vit5-vimedaq-medical-qa"
MAX_INPUT   = 1024
MAX_OUTPUT  = 256
CROSS_ENCODER_THRESHOLD = 0.4


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CORPUS_PATH = os.path.join(BASE_DIR, "medical_corpus_V3.json")
FAISS_PATH  = os.path.join(BASE_DIR, "faiss_index_V3.bin")
BM25_PATH   = os.path.join(BASE_DIR, "bm25_index_V3.pkl")

# ===========================================================================
# 2. KHỞI TẠO DỮ LIỆU & INDEX
# ===========================================================================
print("⏳ Đang tải Knowledge Base và các file Index...")
try:
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
        # Tự động trích xuất đúng mảng dữ liệu (List)
        if isinstance(raw_data, dict) and "texts" in raw_data:
            corpus = raw_data["texts"]
        elif isinstance(raw_data, dict): # Trường hợp lưu dạng dict {"0": "text", "1": "text"}
            corpus = [raw_data[str(i)] for i in range(len(raw_data))]
        else:
            corpus = raw_data

    faiss_index = faiss.read_index(FAISS_PATH)
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    print(f"✅ Đã nạp thành công {len(corpus)} tài liệu y tế (FAISS + BM25)")
except Exception as e:
    print(f"⚠️ Lỗi khởi tạo dữ liệu. Hãy đảm bảo các file nằm đúng thư mục. Chi tiết: {e}")
    corpus, faiss_index, bm25 = [], None, None

# ===========================================================================
# 3. KHỞI TẠO MÔ HÌNH (LLM, BI-ENCODER, CROSS-ENCODER)
# ===========================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Hệ thống đang chạy trên: {device.upper()}")

print("🔄 Đang tải SentencePiece Tokenizer thô (chống lỗi KeyError)...")
spiece_path = hf_hub_download(repo_id="VietAI/vit5-base", filename="spiece.model")
sp = spm.SentencePieceProcessor()
sp.Load(spiece_path)

PAD_ID, EOS_ID, UNK_ID = 0, 1, 2

print(f"🔄 Đang nạp mô hình sinh LLM ({MODEL_ID})...")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
model.config.decoder_start_token_id = PAD_ID
if model.generation_config is not None:
    model.generation_config.decoder_start_token_id = PAD_ID
model = model.to(device)
model.eval()

print("🔄 Đang nạp Bi-Encoder (dùng cho FAISS)...")
retriever_model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder", device=device)

print("🔄 Đang nạp Cross-Encoder (mMarco)...")
cross_encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device=device)

# ===========================================================================
# 4. HÀM PHỤ TRỢ TOKENIZER & BỘ LỌC ẢO GIÁC
# ===========================================================================
def sp_encode(text, max_length=512):
    ids = sp.EncodeAsIds(text)
    if len(ids) > max_length - 1:
        ids = ids[:max_length - 1]
    ids.append(EOS_ID) 
    return ids

def sp_decode(ids):
    clean_ids = [i for i in ids if i not in (PAD_ID, EOS_ID, UNK_ID)]
    return sp.DecodeIds(clean_ids)

# ===========================================================================
# 5. LUỒNG TRUY XUẤT 3 BƯỚC (FAISS + BM25 -> RRF -> CROSS-ENCODER)
# ===========================================================================
def retrieve_contexts(question: str, top_k: int = 3) -> list[str]:
    if not corpus: return []

    # Chuyển câu hỏi về chữ thường để BM25 không bị mù
    search_query = question.lower().strip()
    
    search_query_expanded = search_query
    # STAGE 1a: FAISS Search (Ngữ nghĩa) - Đưa câu hỏi mở rộng vào
    query_emb = retriever_model.encode([search_query_expanded], normalize_embeddings=True, show_progress_bar=False)
    query_emb_f32 = np.array(query_emb).astype('float32')
    distances, faiss_indices = faiss_index.search(query_emb_f32, 50)
    
    dense_rank_dict = {idx: rank for rank, idx in enumerate(faiss_indices[0]) if 0 <= idx < len(corpus)}

    # STAGE 1b: BM25 Search (Từ khóa) - Đưa câu hỏi đã .lower() vào
    tokenized_query = ViTokenizer.tokenize(search_query).split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranking = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:50]
    
    # Thêm điều kiện (if 0 <= idx < len(corpus)) để chống lỗi out-of-bounds
    bm25_rank_dict = {idx: rank for rank, idx in enumerate(bm25_ranking) if 0 <= idx < len(corpus)}

    # STAGE 2: RRF Fusion (Trộn FAISS & BM25)
    rrf_scores = {}
    RRF_K = 50
    weight_bm25 = 0.35 
    weight_faiss = 0.65 
    
    all_candidates = set(dense_rank_dict.keys()).union(set(bm25_rank_dict.keys()))
    
    for doc_idx in all_candidates:
        faiss_rank = dense_rank_dict.get(doc_idx, 1000) 
        bm25_rank = bm25_rank_dict.get(doc_idx, 1000)
        score = (weight_bm25 / (RRF_K + bm25_rank)) + (weight_faiss / (RRF_K + faiss_rank))
        rrf_scores[doc_idx] = score
        
    top_20_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:50]
    candidate_indices = [idx for idx, _ in top_20_candidates]

    # STAGE 3: Cross-Encoder (Rerank)
    # Phải dùng câu hỏi mở rộng để Cross-Encoder đo độ liên quan chính xác
    cross_inp = [[search_query_expanded, corpus[idx]] for idx in candidate_indices]
    cross_scores = cross_encoder.predict(cross_inp, batch_size=20)
    
    reranked_order = sorted(range(len(cross_scores)), key=lambda k: cross_scores[k], reverse=True)
    
    best_ce_score = cross_scores[reranked_order[0]]
    prob_score = 1 / (1 + math.exp(-best_ce_score))
    
    if prob_score < CROSS_ENCODER_THRESHOLD:
        print(f"[CẢNH BÁO] Hệ thống từ chối do độ tin cậy thấp: {prob_score:.4f}")
        return []

    best_indices = [candidate_indices[i] for i in reranked_order[:top_k]]
    return [corpus[idx] for idx in best_indices]
# ===========================================================================
# 6. LUỒNG SINH CÂU TRẢ LỜI ĐẦU CUỐI
# ===========================================================================
def answer_question(question: str, num_beams: int = 2, max_new_tokens: int = 128, top_k: int = 3) -> tuple[str, str]:
    # Quản lý RAM chặt chẽ
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    if not question.strip():
        return "⚠️ Vui lòng nhập câu hỏi y tế.", ""

    contexts = retrieve_contexts(question, top_k=top_k)
    
    if not contexts:
        return "Xin lỗi, dữ liệu y khoa đáng tin cậy của hệ thống không có thông tin chính xác về vấn đề này. Bạn không nên tự ý dùng thuốc mà hãy tham khảo ý kiến bác sĩ chuyên khoa.", "🚨 Đã chặn truy xuất rác."

    contexts_display = "\n\n".join([f"--- Context {i+1} ---\n{ctx}" for i, ctx in enumerate(contexts)])
    combined_context = " . ".join(contexts)
    input_text = f"question: {question} context: {combined_context}"

    input_ids = sp_encode(input_text, max_length=MAX_INPUT)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask = torch.ones_like(input_tensor).to(device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=50,
                length_penalty=2,
                num_beams=num_beams,
                early_stopping=True if num_beams > 1 else False,
                decoder_start_token_id=PAD_ID,
                repetition_penalty=1.5,
                use_cache=True,
            )
        output_ids = outputs[0].tolist()
        answer = sp_decode(output_ids).strip()
    finally:
        del input_tensor, attention_mask, outputs
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if not answer:
        answer = "Xin lỗi, tôi không tìm được câu trả lời phù hợp dựa trên ngữ cảnh hiện tại."

    #answer = post_process_hallucination(question, answer)
    return answer, contexts_display

# ===========================================================================
# 7. GIAO DIỆN HUGGING FACE SPACES
# ===========================================================================
custom_css = """
.header-container {
    text-align: center; margin-bottom: 25px;
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.08), rgba(16, 185, 129, 0.08));
    padding: 35px 20px; border-radius: 16px; border: 1px solid rgba(6, 182, 212, 0.15);
}
.header-title {
    font-family: 'Outfit', 'Inter', sans-serif !important;
    background: linear-gradient(135deg, #0ea5e9, #10b981);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 800 !important; font-size: 2.4rem !important; margin-bottom: 12px !important;
}
.header-subtitle { color: #4b5563; font-size: 1.15rem; max-width: 800px; margin: 0 auto; line-height: 1.6; }
.submit-btn {
    background: linear-gradient(135deg, #0ea5e9, #10b981) !important; color: white !important;
    font-weight: 600 !important; border: none !important; transition: all 0.25s ease !important;
}
.submit-btn:hover { transform: translateY(-2px) !important; filter: brightness(1.05); }
.accordion-settings { border: 1px solid rgba(209, 213, 219, 0.6) !important; border-radius: 12px !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("""
        <div class="header-container">
            <h1 class="header-title">🏥 ViMedAQA — Trợ lý Ảo Y tế Tiếng Việt</h1>
            <p class="header-subtitle">
                Hệ thống hỏi đáp y khoa tự động kết hợp truy xuất đa tầng (FAISS + BM25 + Cross-Encoder) 
                và sinh câu trả lời bằng mô hình <strong>ViT5-base</strong>.
            </p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="Câu hỏi y tế (Question)",
                placeholder="Nhập câu hỏi y tế bằng tiếng Việt...",
                lines=3
            )
            with gr.Row():
                clear_btn = gr.Button("🗑️ Xóa", variant="secondary")
                submit_btn = gr.Button("⚡ Gửi câu hỏi", variant="primary", elem_classes="submit-btn")
            
            gr.Examples(
                examples=[
                    ["Triệu chứng của bệnh tiểu đường là gì?"],
                    ["Thuốc Paracetamol có tác dụng gì?"],
                    ["Viêm gan B lây qua đường nào?"],
                    ["Sốt xuất huyết có nguy hiểm không"]
                ],
                inputs=[question_input], label="💡 Câu hỏi ví dụ gợi ý"
            )
            
        with gr.Column(scale=2):
            with gr.Accordion("⚙️ Cấu hình Tối ưu sinh văn bản", open=True, elem_classes="accordion-settings"):
                num_beams_slider = gr.Slider(1, 4, value=2, step=1, label="Beam Search (num_beams)", info="Khuyên dùng 2 cho cân bằng tốc độ/chất lượng.")
                max_tokens_slider = gr.Slider(32, 1024, value=128, step=8, label="Độ dài câu trả lời tối đa")
                top_k_slider = gr.Slider(1, 5, value=3, step=1, label="Số văn bản truy xuất (Top-K Contexts)")
                
    with gr.Row():
        with gr.Column(scale=1):
            answer_output = gr.Textbox(label="🏥 Câu trả lời từ Trợ lý Ảo", lines=6, interactive=False)
        with gr.Column(scale=1):
            context_output = gr.Textbox(label="📄 Ngữ cảnh y tế tham khảo", lines=8, interactive=False)

    submit_btn.click(fn=answer_question, inputs=[question_input, num_beams_slider, max_tokens_slider, top_k_slider], outputs=[answer_output, context_output])
    question_input.submit(fn=answer_question, inputs=[question_input, num_beams_slider, max_tokens_slider, top_k_slider], outputs=[answer_output, context_output])
    clear_btn.click(fn=lambda: ("", "", ""), inputs=[], outputs=[question_input, answer_output, context_output])

if __name__ == "__main__":
    demo.launch(share=False)