# 🏥 ViMedQA RAG: Trợ Lý Y Tế Thông Minh Tiếng Việt

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Model-ViT5--Base-orange.svg)](https://huggingface.co/VietAI/vit5-base)
[![Course](https://img.shields.io/badge/M%C3%B4n%20h%E1%BB%8Dc-H%E1%BB%8Dc%20Th%E1%BB%91ng%20K%C3%AA-green.svg)](#)

Dự án xây dựng hệ thống Trợ Lý Y Tế ảo (Medical Virtual Assistant) dành riêng cho người dùng Việt Nam, ứng dụng kỹ thuật **RAG (Retrieval-Augmented Generation)** kết hợp mô hình ngôn ngữ **ViT5** fine-tuned. Hệ thống cung cấp câu trả lời có trích dẫn nguồn, giảm thiểu thông tin sai lệch (hallucination) và tích hợp cảnh báo an toàn y tế.

---

## 👥 Nhóm Thực Hiện

**Môn học:** Học Thống Kê

| STT | Họ và Tên             |
| :-: | :-------------------- |
|  1  | **Nguyễn Kim Quốc**   |
|  2  | **Nguyễn Tiến Thành** |
|  3  | **Phạm Minh Thông**   |

---

## 🚀 Tính Năng Chính

- **Trả lời câu hỏi y tế:** Giải đáp các thắc mắc về triệu chứng, bệnh lý và sức khỏe bằng tiếng Việt.
- **Truy vấn thông minh (RAG):** Tìm kiếm và tổng hợp thông tin từ cơ sở dữ liệu y khoa đáng tin cậy.
- **Trích dẫn nguồn (Citations):** Hiển thị rõ ràng nguồn thông tin tham khảo cho mỗi câu trả lời.
- **Cảnh báo an toàn:** Tự động đưa ra các disclaimer và khuyến cáo người dùng thăm khám bác sĩ.
- **Giao diện Doctor Chatbot:** Trải nghiệm chat chuyên nghiệp, thân thiện và trực quan.

---

## 🛠️ Công Nghệ Sử Dụng

- **Ngôn ngữ:** Python 3.10+
- **Deep Learning:** PyTorch, HuggingFace Transformers
- **Mô hình sinh (Generator):** `VietAI/vit5-base` (270M parameters)
- **Embedding & Retrieval:** `vinai/phobert-base-v2`, FAISS Vector Store
- **Giao diện:** Streamlit 1.x
- **Môi trường:** Google Colab / Kaggle (Training), Docker (Deployment)

---

## 📊 Dữ Liệu & Tiền Xử Lý

### 1. Nguồn dữ liệu

Dự án sử dụng kết hợp các bộ dữ liệu y khoa tiếng Việt chất lượng cao:

- **ViMedAQA (ACL 2024):** Khoảng 2,000+ mẫu câu hỏi và trả lời y khoa được kiểm duyệt.
- **Vietnamese-Medical-QA:** 9,335 mẫu dữ liệu từ các nền tảng eDoctor và Vinmec.

### 2. Quy trình xử lý (preprocess.py)

Dữ liệu được đi qua pipeline làm sạch nghiêm ngặt:

1. **Làm sạch nhiễu:** Loại bỏ lời chào, quảng cáo và các ký tự thừa bằng Regex.
2. **Chuẩn hóa:** Đưa về chuẩn Unicode (NFC) và xử lý dấu câu tiếng Việt.
3. **Bảo mật:** Loại bỏ các thông tin định danh cá nhân (PII) để đảm bảo quyền riêng tư.
4. **Phân tách từ:** Sử dụng `VnCoreNLP` để thực hiện word segmentation.
5. **Phân chia dữ liệu:** Chia tập Train/Val/Test theo tỉ lệ 80/10/10, đảm bảo cân bằng theo chủ đề (stratified).

---

## 📂 Cấu Trúc Thư Mục

```text
.
├── app/                # Giao diện Streamlit và components
├── configs/            # File cấu hình (hyperparameters, paths)
├── data/
│   ├── raw/            # Dataset gốc chưa xử lý
│   └── processed/      # Dữ liệu sau khi clean và segment
├── notebooks/          # EDA, Training và Evaluation notebooks
├── src/
│   ├── model/          # Logic huấn luyện và suy luận
│   └── retriever/      # Logic encode và FAISS store
├── docker/             # Dockerfile và docker-compose
└── README.md           # Hướng dẫn dự án
```

---

## 💻 Hướng Dẫn Cài Đặt & Chạy

### 1. Cài đặt môi trường

```bash
# Clone repository
git clone https://github.com/your-repo/VietNamese_MediCal_Virtual_Assistant.git
cd VietNamese_MediCal_Virtual_Assistant

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Xây dựng Vector Store

```bash
python src/retriever/faiss_store.py --build
```

### 3. Chạy ứng dụng Demo

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Đánh Giá Mô Hình

Dự án được đánh giá dựa trên các chỉ số tiêu chuẩn trong NLP:

- **ROUGE-1/2/L:** Đo lường mức độ tương đồng n-gram.
- **BERTScore F1:** Đánh giá ngữ nghĩa sâu của câu trả lời.
- **Latency:** Đảm bảo thời gian phản hồi dưới 3 giây.

---

## ⚠️ Tuyên Bố Miễn Trừ Trách Nhiệm (Disclaimer)

Hệ thống này chỉ mang tính chất tham khảo và hỗ trợ thông tin. **Tuyệt đối không được coi là lời khuyên y tế chuyên nghiệp.** Người dùng nên tham khảo ý kiến của bác sĩ hoặc chuyên gia y tế trước khi đưa ra bất kỳ quyết định nào liên quan đến sức khỏe.

---

© 2025 ViMedQA Project Team. All rights reserved.
