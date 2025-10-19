import os
import pickle
import numpy as np
from PIL import Image
import streamlit as st  # <-- Thư viện chính của Streamlit

# --- Import các thư viện HOG ---
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import hog

# ---------------- CONFIG HOG (Phải giống hệt file train) ----------------
HOG_IMG_SIZE = (128, 64)
PIXELS_PER_CELL = (16, 16)
CELLS_PER_BLOCK = (2, 2)
ORIENTATIONS = 9

# ---------------- Tải Model đã huấn luyện ----------------
MODEL_PATH = os.path.join("outputs", "softmax_model_hog_improved.pkl")


# Dùng cache của Streamlit để chỉ tải model một lần
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        # Hiển thị lỗi trên giao diện Streamlit
        st.error(f"Lỗi: Không tìm thấy file model tại '{MODEL_PATH}'")
        st.stop()  # Dừng ứng dụng

    print(f"Đang tải model từ {MODEL_PATH}...")
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    print("Tải model thành công.")
    return model_data


model_data = load_model()

# Lấy các thành phần từ model
W = model_data["W"]
b = model_data["b"]
mean = model_data["mean"]
std = model_data["std"]
label_map = model_data["label_map"]

# Tạo map ngược
inv_label_map = {v: k for k, v in label_map.items()}


# ---------------- Hàm tính Softmax (Lấy từ file train) ----------------
def softmax_np(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


# ---------------- Hàm trích xuất đặc trưng HOG (Lấy từ file train) ----------------
def extract_hog_features(img_pil):
    try:
        img = np.array(img_pil)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]  # Loại bỏ kênh Alpha

        resized_img = resize(img, HOG_IMG_SIZE, anti_aliasing=True)
        gray_img = rgb2gray(resized_img) if resized_img.ndim == 3 else resized_img

        features = hog(gray_img, orientations=ORIENTATIONS,
                       pixels_per_cell=PIXELS_PER_CELL,
                       cells_per_block=CELLS_PER_BLOCK,
                       block_norm='L2-Hys',
                       visualize=False,
                       transform_sqrt=True)
        return features
    except Exception as e:
        print(f"Lỗi khi trích xuất HOG: {e}")
        return None


# ---------------- Xây dựng giao diện Streamlit ----------------

st.title("Phân loại ảnh điện thoại (HOG + Softmax)")

# 1. Tạo nút tải file
uploaded_file = st.file_uploader("Chọn một ảnh để dự đoán:",
                                 type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # 2. Hiển thị ảnh
    img_pil = Image.open(uploaded_file)
    st.image(img_pil, caption="Ảnh đã tải lên", use_column_width=True)

    # 3. Tạo nút dự đoán
    if st.button("Dự đoán"):
        # 4. Xử lý và dự đoán
        with st.spinner("Đang trích xuất đặc trưng HOG và dự đoán..."):
            features = extract_hog_features(img_pil)

            if features is None:
                st.error("Không thể xử lý ảnh này.")
            else:
                # Chuẩn hóa đặc trưng
                features_2d = features.reshape(1, -1)
                features_std = (features_2d - mean) / (std + 1e-12)

                # Dự đoán
                scores = features_std @ W + b
                probs = softmax_np(scores)

                pred_index = np.argmax(probs, axis=1)[0]
                prediction_label = inv_label_map[pred_index]
                probability = np.max(probs) * 100

                # 5. Hiển thị kết quả
                st.success(f"**Kết quả:** '{prediction_label}'")
                st.info(f"**Độ tin cậy:** {probability:.2f}%")