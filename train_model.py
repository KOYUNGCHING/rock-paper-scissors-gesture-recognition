# ==============================
# train_model.py
# 功能：
# 使用自己收集的剪刀石頭布 landmark CSV 訓練 TensorFlow 手勢分類模型
#
# 輸入：
# gesture_landmarks_custom.csv
#
# 輸出：
# 1. gesture_tf_model.keras
# 2. label_encoder.pkl
# 3. training_accuracy.png
# 4. training_loss.png
# 5. confusion_matrix.png
# ==============================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# ==============================
# 1. 讀取自己的 CSV 資料
# ==============================

csv_path = Path("gesture_landmarks_custom.csv")

if not csv_path.exists():
    raise FileNotFoundError(
        "找不到 gesture_landmarks_custom.csv。"
        "請先執行 python create_custom_dataset.py 產生資料集。"
    )

df = pd.read_csv(csv_path)

# 清除欄位名稱前後空白
df.columns = df.columns.str.strip()

# 清除 label 前後空白
df["gesture_label"] = df["gesture_label"].astype(str).str.strip()

print("\nData shape:", df.shape)
print("Columns:", df.columns[:5].tolist(), "...", df.columns[-5:].tolist())

print("\nGesture label counts:")
print(df["gesture_label"].value_counts())


# ==============================
# 2. 檢查資料是否足夠
# ==============================

if df.empty:
    raise ValueError("CSV 是空的，代表沒有成功從圖片偵測到手。請檢查圖片或 create_custom_dataset.py。")

num_classes = df["gesture_label"].nunique()

if num_classes < 2:
    raise ValueError("至少需要 2 個以上的手勢類別才能訓練模型。")

min_count = df["gesture_label"].value_counts().min()

if min_count < 5:
    print("\n警告：某些類別資料少於 5 筆，模型可能非常不穩。")


# ==============================
# 3. 建立 X 和 y
# ==============================

X = df.drop(columns=["gesture_label"]).values.astype(np.float32)
y = df["gesture_label"].values

print("\nX shape:", X.shape)
print("y shape:", y.shape)
print("Labels:", np.unique(y))


# ==============================
# 4. Label Encoding
# ==============================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nLabel classes:")
print(label_encoder.classes_)


# ==============================
# 5. 切分 train / test
# ==============================
# 如果資料太少，test_size 可以改小一點，例如 0.15。
# 目前各類 30 張左右，用 0.2 可以接受。

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTrain/Test split:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


# ==============================
# 6. 建立 TensorFlow 模型
# ==============================

num_classes = len(label_encoder.classes_)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==============================
# 7. 訓練模型
# ==============================
# 你的資料目前比較少，所以模型不要太大。
# epochs 可以先設 80。
# batch_size=8 對小資料比較適合。

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=80,
    batch_size=8,
    verbose=1
)


# ==============================
# 8. 測試模型
# ==============================

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_acc)


# ==============================
# 9. Classification Report
# ==============================

y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))


# ==============================
# 10. 畫 accuracy / loss 圖
# ==============================

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("training_accuracy.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.tight_layout()
plt.savefig("training_loss.png", dpi=300)
plt.show()


# ==============================
# 11. Confusion Matrix
# ==============================

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()


# ==============================
# 12. 儲存模型與 Label Encoder
# ==============================

model.save("gesture_tf_model.keras")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\nSaved files:")
print("- gesture_tf_model.keras")
print("- label_encoder.pkl")
print("- training_accuracy.png")
print("- training_loss.png")
print("- confusion_matrix.png")