import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import multiprocessing
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import hog

# ---------------- CONFIG ----------------
DATA_ROOT = r"D:\HOMEWORK\dataset_1\dataset"  # <-- Thay đổi đường dẫn tới thư mục dataset của bạn
SPLITS = ["train\phone", "val\phone", "test\phone"]

LABEL_MAP = {
    "non-phone": 0,
    "defective": 1,
    "non-defective": 2
}

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# --- THAY ĐỔI: Cấu hình cho HOG ---
HOG_IMG_SIZE = (128, 64)
PIXELS_PER_CELL = (16, 16)
CELLS_PER_BLOCK = (2, 2)
ORIENTATIONS = 9

DO_AUGMENT = True
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# tham số cho Softmax
LR = 0.005
LAMBDA = 1 # Regularization mạnh hơn
EPOCHS = 200 # Số epoch tối đa
EARLY_STOPPING_PATIENCE = 15

# ---------------- Worker Function for Multiprocessing ----------------
def _worker_process_hog(args):
    path, do_flip = args
    try:
        img_pil = Image.open(path)
        if do_flip:
            img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)

        img = np.array(img_pil)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3] # Loại bỏ kênh Alpha

        resized_img = resize(img, HOG_IMG_SIZE, anti_aliasing=True)
        gray_img = rgb2gray(resized_img) if resized_img.ndim == 3 else resized_img

        # Trích xuất đặc trưng HOG
        features = hog(gray_img, orientations=ORIENTATIONS,
                       pixels_per_cell=PIXELS_PER_CELL,
                       cells_per_block=CELLS_PER_BLOCK,
                       block_norm='L2-Hys',
                       visualize=False,
                       transform_sqrt = True)
        return features
    except Exception:
        return None

# ---------------- Optimized Feature Extraction ----------------
def build_dataset_for_split(split):
    split_dir = os.path.join(DATA_ROOT, split)
    class_dirs = find_class_dirs(split_dir)

    X_list, y_list = [], []
    for class_name, class_folder in class_dirs:
        label = LABEL_MAP[class_name]
        print(f"Processing class '{class_name}' -> label {label}")
        filepaths = list_images_in_dir(class_folder)
        if not filepaths: continue

        tasks = [(path, False) for path in filepaths]
        if DO_AUGMENT:
            tasks.extend([(path, True) for path in filepaths])

        num_cores = multiprocessing.cpu_count() - 1 or 1
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = list(tqdm(pool.imap(_worker_process_hog, tasks), total=len(tasks), desc="Extracting HOG features"))

        # Thu thập kết quả và tạo nhãn tương ứng
        all_feats = [res for res in results if res is not None]
        if all_feats:
            # Xác định số lượng ảnh gốc và ảnh augment thành công
            orig_tasks_len = len(filepaths)
            num_orig_success = sum(1 for i, res in enumerate(results) if i < orig_tasks_len and res is not None)
            num_aug_success = len(all_feats) - num_orig_success

            y_labels = np.concatenate([
                np.full(num_orig_success, label, dtype=np.int64),
                np.full(num_aug_success, label, dtype=np.int64)
            ])

            X_list.append(np.array(all_feats))
            y_list.append(y_labels)

    if not X_list:
        dummy_img = np.zeros(HOG_IMG_SIZE)
        FEATURE_DIM = hog(dummy_img, orientations=ORIENTATIONS, pixels_per_cell=PIXELS_PER_CELL, cells_per_block=CELLS_PER_BLOCK).shape[0]
        return np.zeros((0, FEATURE_DIM)), np.zeros((0,))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]

# ---------------- Softmax Training with Early Stopping ----------------
def softmax_np(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def compute_loss(X, y, W, b, lambd):
    m = X.shape[0]
    scores = X @ W + b
    probs = softmax_np(scores)
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(m), y] = 1
    ce = -np.sum(y_onehot * np.log(probs + 1e-12)) / m
    reg = (lambd / (2*m)) * np.sum(W**2)
    return ce + reg

def compute_acc(X, y, W, b):
    scores = X @ W + b
    probs = softmax_np(scores)
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == y)

def train_softmax(X_train, y_train, X_val, y_val, X_test, y_test, lr, lambd, epochs, patience):
    m, n = X_train.shape
    n_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))
    W = np.zeros((n, n_classes), dtype=np.float32)
    b = np.zeros((n_classes,), dtype=np.float32)

    best_val_loss = float('inf')
    patience_counter = 0
    best_W, best_b = W.copy(), b.copy() # Khởi tạo best model

    for epoch in range(1, epochs+1):
        scores = X_train @ W + b
        probs = softmax_np(scores)
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(m), y_train] = 1
        dW = (X_train.T @ (probs - y_onehot)) / m + (lambd/m) * W
        db = np.mean(probs - y_onehot, axis=0)
        W -= lr * dW
        b -= lr * db

        if epoch % 5 == 0:
            val_loss = compute_loss(X_val, y_val, W, b, lambd)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_W, best_b = W.copy(), b.copy()
                patience_counter = 0
                print(f"Epoch {epoch} | New best val_loss: {val_loss:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"--- Early stopping triggered at epoch {epoch} ---")
                break

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            current_val_loss = compute_loss(X_val, y_val, W, b, lambd)
            train_loss = compute_loss(X_train, y_train, W, b, lambd)
            test_loss  = compute_loss(X_test, y_test, W, b, lambd)
            train_acc = compute_acc(X_train, y_train, W, b)
            val_acc   = compute_acc(X_val, y_val, W, b)
            test_acc  = compute_acc(X_test, y_test, W, b)
            print(f"Epoch {epoch} | TrainLoss={train_loss:.4f}, ValLoss={current_val_loss:.4f} (Best: {best_val_loss:.4f}), TestLoss={test_loss:.4f} | "
                  f"TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}, TestAcc={test_acc:.4f}")

    return best_W, best_b

# --- Helpers (Không thay đổi) ---
def plot_confusion_matrix(y_true, y_pred, classes=None, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.show()

def find_class_dirs(root_split_dir):
    found = []
    for dirpath, dirnames, filenames in os.walk(root_split_dir):
        basename = os.path.basename(dirpath)
        if basename in LABEL_MAP:
            found.append((basename, dirpath))
    return found

def list_images_in_dir(dir_path):
    filepaths = []
    if not os.path.isdir(dir_path):
        return filepaths
    for root, _, filenames in os.walk(dir_path):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMG_EXTS:
                filepaths.append(os.path.join(root, fname))
    return filepaths

# ---------------- Main ----------------
if __name__ == "__main__":
    feats = {}
    for split in SPLITS:
        safe_split_name = split.replace(os.path.sep, '_')
        out_x = os.path.join(OUT_DIR, f"X_{safe_split_name}_hog_imp.npy")
        out_y = os.path.join(OUT_DIR, f"y_{safe_split_name}_hog_imp.npy")
        if os.path.exists(out_x) and os.path.exists(out_y):
            print(f"[{split}] loading cached HOG features...")
            X = np.load(out_x)
            y = np.load(out_y)
        else:
            print(f"[{split}] extracting HOG features...")
            X, y = build_dataset_for_split(split)
            if X.size > 0:
                np.save(out_x, X)
                np.save(out_y, y)
        print(f"  shapes X={X.shape}, y={y.shape}")
        feats[split] = (X, y)

    X_train, y_train = feats["train\phone"]
    X_val, y_val     = feats["val\phone"]
    X_test, y_test   = feats["test\phone"]

    if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
        print("One of the data splits is empty. Exiting.")
    else:
        mean = X_train.mean(axis=0, keepdims=True)
        std  = X_train.std(axis=0, keepdims=True) + 1e-12
        X_train_std = (X_train - mean) / std
        X_val_std   = (X_val - mean) / std
        X_test_std  = (X_test - mean) / std

        W, b = train_softmax(X_train_std.astype(np.float32), y_train.astype(np.int64),
                            X_val_std.astype(np.float32), y_val.astype(np.int64),
                            X_test_std.astype(np.float32), y_test.astype(np.int64),
                            lr=LR, lambd=LAMBDA, epochs=EPOCHS, patience=EARLY_STOPPING_PATIENCE)

        model_dict = {"W": W, "b": b, "mean": mean, "std": std, "label_map": LABEL_MAP}
        model_path = os.path.join(OUT_DIR, "softmax_model_hog_improved.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model_dict, f)
        print(f"Saved Improved HOG softmax model to {model_path}")

        preds_train = np.argmax(softmax_np(X_train_std @ W + b), axis=1)
        preds_val   = np.argmax(softmax_np(X_val_std @ W + b), axis=1)
        preds_test  = np.argmax(softmax_np(X_test_std @ W + b), axis=1)

        class_names = [k for k, v in sorted(LABEL_MAP.items(), key=lambda x: x[1])]
        plot_confusion_matrix(y_train, preds_train, classes=class_names, title="HOG (Improved) Train Confusion Matrix")
        plot_confusion_matrix(y_val, preds_val, classes=class_names, title="HOG (Improved) Val Confusion Matrix")
        plot_confusion_matrix(y_test, preds_test, classes=class_names, title="HOG (Improved) Test Confusion Matrix")

        print("Done.")
