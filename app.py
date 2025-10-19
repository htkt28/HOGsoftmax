import os
import pickle
import numpy as np
from PIL import Image
from flask import Flask, request, render_template_string

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

# Kiểm tra xem model có tồn tại không
if not os.path.exists(MODEL_PATH):
    print(f"Lỗi: Không tìm thấy file model tại '{MODEL_PATH}'")
    print("Vui lòng chạy file huấn luyện trước để tạo model.")
    exit()

print(f"Đang tải model từ {MODEL_PATH}...")
with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

# Lấy các thành phần từ model
W = model_data["W"]
b = model_data["b"]
mean = model_data["mean"]
std = model_data["std"]
label_map = model_data["label_map"]

# Tạo map ngược để lấy tên class từ index
# Ví dụ: {0: 'non-phone', 1: 'defective', 2: 'non-defective'}
inv_label_map = {v: k for k, v in label_map.items()}

print("Tải model thành công.")

# ---------------- Khởi tạo ứng dụng Flask ----------------
app = Flask(__name__)


# ---------------- Hàm tính Softmax (Lấy từ file train) ----------------
def softmax_np(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


# ---------------- Hàm trích xuất đặc trưng HOG (Lấy từ file train) ----------------
def extract_hog_features(img_pil):
    """
    Trích xuất đặc trưng HOG từ một đối tượng PIL Image.
    Trả về một vector 1D (numpy array).
    """
    try:
        img = np.array(img_pil)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]  # Loại bỏ kênh Alpha

        resized_img = resize(img, HOG_IMG_SIZE, anti_aliasing=True)
        gray_img = rgb2gray(resized_img) if resized_img.ndim == 3 else resized_img

        # Trích xuất đặc trưng HOG
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


# ---------------- Định nghĩa trang Web ----------------

# HTML cho trang chủ (có form upload và hiển thị kết quả)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dự đoán lỗi điện thoại</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f4f4f4; }
        .container { max-width: 600px; margin: auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #333; }
        form { margin-top: 20px; }
        input[type="file"] { display: block; margin-bottom: 10px; }
        input[type="submit"] { background: #007bff; color: #fff; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; }
        input[type="submit"]:hover { background: #0056b3; }
        #result { margin-top: 30px; padding: 15px; border-radius: 4px; }
        .res-success { background: #d4edda; border-color: #c3e6cb; color: #155724; }
        .res-fail { background: #f8d7da; border-color: #f5c6cb; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Phân loại ảnh điện thoại (HOG + Softmax)</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <label for="file">Chọn một ảnh để dự đoán:</label>
            <input type="file" name="file" id="file" accept="image/*" required>
            <input type="submit" value="Dự đoán">
        </form>

        {% if prediction %}
            <div id="result" class="res-success">
                <strong>Kết quả:</strong> {{ prediction }}
            </div>
        {% endif %}

        {% if error %}
            <div id="result" class="res-fail">
                <strong>Lỗi:</strong> {{ error }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route('/', methods=['GET'])
def home():
    """Hiển thị trang chủ."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    """Xử lý ảnh tải lên và trả về dự đoán."""
    if 'file' not in request.files:
        return render_template_string(HTML_TEMPLATE, error="Không có file nào được tải lên.")

    file = request.files['file']

    if file.filename == '':
        return render_template_string(HTML_TEMPLATE, error="Chưa chọn file.")

    if file:
        try:
            # 1. Mở ảnh
            img_pil = Image.open(file.stream)

            # 2. Trích xuất đặc trưng HOG
            features = extract_hog_features(img_pil)
            if features is None:
                return render_template_string(HTML_TEMPLATE, error="Không thể xử lý ảnh.")

            # 3. Chuẩn hóa đặc trưng (Standardize)
            # features là 1D (756,), cần reshape thành 2D (1, 756)
            features_2d = features.reshape(1, -1)
            features_std = (features_2d - mean) / (std + 1e-12)  # Thêm 1e-12 để tránh chia cho 0

            # 4. Dự đoán
            scores = features_std @ W + b  # (1, 756) @ (756, 3) + (3,) -> (1, 3)
            probs = softmax_np(scores)  # (1, 3)

            # Lấy index có xác suất cao nhất
            pred_index = np.argmax(probs, axis=1)[0]

            # 5. Lấy tên class
            prediction_label = inv_label_map[pred_index]

            # Trả về kết quả
            return render_template_string(HTML_TEMPLATE,
                                          prediction=f"'{prediction_label}' (Xác suất: {np.max(probs) * 100:.2f}%)")

        except Exception as e:
            return render_template_string(HTML_TEMPLATE, error=f"Đã xảy ra lỗi: {e}")


# ---------------- Chạy ứng dụng ----------------
if __name__ == "__main__":
    print("Khởi chạy Flask server tại http://127.0.0.1:5000")
    app.run(debug=True, port=5000)