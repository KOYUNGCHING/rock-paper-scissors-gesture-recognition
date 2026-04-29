from pathlib import Path

# 要處理的資料夾，也就是三個手勢類別
classes = ["paper", "rock", "scissors"]

# 支援的圖片副檔名
image_extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

for class_name in classes:
    folder = Path(class_name)

    if not folder.exists():
        print(f"找不到資料夾：{folder}")
        continue

    # 找出該資料夾內所有圖片
    images = []
    for ext in image_extensions:
        images.extend(folder.glob(f"*{ext}"))

    # 排序，讓命名順序固定
    images = sorted(images)

    print(f"{class_name}: 找到 {len(images)} 張圖片")

    for idx, image_path in enumerate(images, start=1):
        new_name = f"{class_name}_{idx:03d}{image_path.suffix.lower()}"
        new_path = folder / new_name

        # 避免原本檔名剛好一樣時出錯
        if image_path != new_path:
            image_path.rename(new_path)

    print(f"{class_name}: 重新命名完成")

print("全部圖片重新命名完成！")