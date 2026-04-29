# ==============================
# create_custom_dataset.py
# 功能：
# 讀取 paper / rock / scissors 資料夾中的圖片
# 使用 MediaPipe HandLandmarker 偵測手部 landmarks
# 將 landmarks 轉成 63 維特徵
# 存成 gesture_landmarks_custom.csv
# ==============================

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==============================
# 1. 基本設定
# ==============================

# 你的三個手勢類別，資料夾名稱要和這裡一樣
CLASSES = ["paper", "rock", "scissors"]

# 輸出的 CSV 檔名
OUTPUT_CSV = "gesture_landmarks_custom.csv"

# MediaPipe hand landmarker model
TASK_MODEL_PATH = "hand_landmarker.task"

# 支援的圖片格式
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


# ==============================
# 2. 建立 MediaPipe Hand Landmarker
# ==============================

base_options = python.BaseOptions(
    model_asset_path=TASK_MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)


# ==============================
# 3. 建立欄位名稱
# ==============================

columns = []

for i in range(21):
    columns.extend([
        f"landmark_{i}_x",
        f"landmark_{i}_y",
        f"landmark_{i}_z"
    ])

columns.append("gesture_label")


# ==============================
# 4. 處理每一張圖片
# ==============================

rows = []

total_images = 0
success_images = 0
failed_images = 0

for class_name in CLASSES:
    folder = Path(class_name)

    if not folder.exists():
        print(f"找不到資料夾：{folder}")
        continue

    image_paths = []

    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(folder.glob(f"*{ext}"))

    image_paths = sorted(image_paths)

    print(f"\n處理類別：{class_name}")
    print(f"找到圖片數量：{len(image_paths)}")

    for image_path in image_paths:
        total_images += 1

        # 讀取圖片
        image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            print(f"無法讀取圖片：{image_path}")
            failed_images += 1
            continue

        # OpenCV 是 BGR，MediaPipe 使用 RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 建立 MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )

        # 偵測手部 landmarks
        detection_result = detector.detect(mp_image)

        if not detection_result.hand_landmarks:
            print(f"沒有偵測到手：{image_path}")
            failed_images += 1
            continue

        # 只取第一隻手
        hand_landmarks = detection_result.hand_landmarks[0]

        # ==============================
        # 5. landmark normalization
        # ==============================
        # 這裡用 landmark 0，也就是 wrist，當作原點
        # 每個點都減掉 wrist
        # 再用 wrist 到 middle MCP landmark 9 的距離做尺度標準化
        # 這樣可以降低手離鏡頭遠近不同造成的影響

        wrist = hand_landmarks[0]

        coords = []

        for lm in hand_landmarks:
            coords.append([
                lm.x - wrist.x,
                lm.y - wrist.y,
                lm.z - wrist.z
            ])

        coords = np.array(coords, dtype=np.float32)

        # 用 landmark 9 到 wrist 的距離當作手掌大小
        scale = np.linalg.norm(coords[9])

        if scale < 1e-6:
            scale = 1.0

        coords = coords / scale

        # 攤平成 63 維
        features = coords.flatten().tolist()

        # 加上 label
        features.append(class_name)

        rows.append(features)
        success_images += 1

print("\n==============================")
print("資料轉換完成")
print("==============================")
print("總圖片數量:", total_images)
print("成功偵測:", success_images)
print("失敗圖片:", failed_images)


# ==============================
# 6. 存成 CSV
# ==============================

df = pd.DataFrame(rows, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n已儲存：{OUTPUT_CSV}")
print("CSV shape:", df.shape)
print("\n各類別數量：")
print(df["gesture_label"].value_counts())