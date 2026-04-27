import streamlit as st
import time

# Cấu hình trang
st.set_page_config(
    page_title="ViMedQA - Trợ Lý Y Tế Thông Minh",
    page_icon="🏥",
    layout="centered"
)

# Giao diện Header
st.title("ViMedQA RAG")
st.markdown("### Trợ Lý Y Tế Thông Minh Tiếng Việt")
st.info("Hệ thống đang sử dụng mô hình ViT5 + RAG để hỗ trợ giải đáp thắc mắc y tế.")

# Sidebar cấu hình
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/387/387561.png", width=100)
    st.header("Cấu hình hệ thống")
    top_k = st.slider("Số lượng văn bản truy vấn (Top-K)", 1, 5, 3)
    st.divider()
    st.warning("**Lưu ý:** Mọi thông tin chỉ mang tính chất tham khảo. Hãy thăm khám bác sĩ để có lời khuyên chính xác nhất.")

# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Khung nhập câu hỏi
if prompt := st.chat_input("Nhập câu hỏi y tế của bạn tại đây..."):
    # Hiển thị câu hỏi của người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Giả lập phản hồi từ mô hình AI
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Đây là nơi sau này bạn sẽ gọi logic từ src/model/inference.py
        assistant_response = f"Chào bạn, cảm ơn bạn đã đặt câu hỏi: '{prompt}'. Hiện tại hệ thống đang trong quá trình kết nối với mô hình ViT5. Đây là câu trả lời mẫu từ trợ lý y tế của bạn."
        
        # Hiệu ứng gõ chữ
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        
        # Hiển thị nguồn trích dẫn (Evidence) mẫu
        with st.expander("Xem nguồn trích dẫn (Citations)"):
            st.write("1. Cổng thông tin Bộ Y Tế Việt Nam")
            st.write("2. Dataset ViMedAQA (ACL 2024)")

    st.session_state.messages.append({"role": "assistant", "content": full_response})
