# ==============================
# predict_image.py
# 功能：輸入一張手勢圖片，輸出手勢分類結果
# 使用新版 MediaPipe Tasks API
# ==============================

import cv2
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==============================
# 1. 載入 TensorFlow 模型與 Label Encoder
# ==============================

model = tf.keras.models.load_model("gesture_tf_model.keras")
label_encoder = joblib.load("label_encoder.pkl")

print("Model and label encoder loaded successfully.")


# ==============================
# 2. 讀取圖片
# ==============================

image_path = "test_paper.jpg"

image_bgr = cv2.imread(image_path)

if image_bgr is None:
    raise FileNotFoundError(f"找不到圖片：{image_path}，請確認圖片有放在同一個資料夾。")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


# ==============================
# 3. 建立新版 MediaPipe Hand Landmarker
# ==============================

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)


# ==============================
# 4. 用 MediaPipe 偵測手部 landmarks
# ==============================

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=image_rgb
)

detection_result = detector.detect(mp_image)

if not detection_result.hand_landmarks:
    print("沒有偵測到手，請換一張更清楚的圖片。")
else:
    print("成功偵測到手。")

    hand_landmarks = detection_result.hand_landmarks[0]

    # ==============================
    # 5. 將 21 個 landmark 轉成 63 維特徵
    # ==============================
    # Kaggle 資料中 landmark_0 是 wrist，通常會被當成原點。
    # 所以這裡也把第 0 點 wrist 當作原點。
    # 每個 landmark 都減掉 wrist 座標，讓資料變成相對座標。

    wrist = hand_landmarks[0]

    feature_list = []

    for lm in hand_landmarks:
        x = lm.x - wrist.x
        y = lm.y - wrist.y
        z = lm.z - wrist.z
        feature_list.extend([x, y, z])

    input_data = np.array(feature_list, dtype=np.float32).reshape(1, 63)

    print("Input feature shape:", input_data.shape)

    # ==============================
    # 6. 用 TensorFlow 模型預測
    # ==============================

    prob = model.predict(input_data)
    pred_index = int(np.argmax(prob, axis=1)[0])

    pred_label = label_encoder.inverse_transform([pred_index])[0]
    confidence = float(np.max(prob))

    print("Predicted gesture:", pred_label)
    print("Confidence:", confidence)

    # ==============================
    # 7. 畫出 landmarks
    # ==============================

    annotated_image = image_rgb.copy()
    h, w, _ = annotated_image.shape

    # 手部骨架連線規則
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),          # index
        (0, 9), (9, 10), (10, 11), (11, 12),     # middle
        (0, 13), (13, 14), (14, 15), (15, 16),   # ring
        (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
        (5, 9), (9, 13), (13, 17)               # palm
    ]

    # 畫線
    for start, end in connections:
        x1 = int(hand_landmarks[start].x * w)
        y1 = int(hand_landmarks[start].y * h)
        x2 = int(hand_landmarks[end].x * w)
        y2 = int(hand_landmarks[end].y * h)

        cv2.line(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 畫點
    for i, lm in enumerate(hand_landmarks):
        x = int(lm.x * w)
        y = int(lm.y * h)

        cv2.circle(annotated_image, (x, y), 4, (255, 0, 0), -1)
        cv2.putText(
            annotated_image,
            str(i),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1
        )

    plt.figure(figsize=(6, 6))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Prediction: {pred_label} ({confidence:.2f})")
    plt.show()