# Sử dụng Python 3.10-slim làm base image
FROM python:3.10-slim

# Thiết lập biến môi trường
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Cài đặt các phụ thuộc hệ thống (bao gồm Java cho VnCoreNLP)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    g++ \
    openjdk-21-jre-headless \
    curl \
    git \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép file requirements.txt
COPY requirements.txt .

# Nâng cấp pip
RUN pip install --no-cache-dir --upgrade pip

# Chia nhỏ cài đặt để dễ gỡ lỗi
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir transformers sentencepiece
RUN pip install --no-cache-dir faiss-cpu datasets py-vncorenlp
RUN pip install --no-cache-dir streamlit pandas numpy pyyaml huggingface_hub tqdm

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Expose cổng mặc định của Streamlit
EXPOSE 8501

# Lệnh chạy ứng dụng khi container khởi động
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
