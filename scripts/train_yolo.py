#!/usr/bin/env python3
"""
训练自定义 YOLOv8 模型（灭火器 + 火焰检测）
使用方法：
  1. 先运行 collect_data.py 采集图片到 ~/yolo_dataset/images/
  2. 用 labelImg 标注后生成 YOLO 格式标签到 ~/yolo_dataset/labels/
  3. 运行此脚本开始训练
"""
from ultralytics import YOLO
import os

# ===== 配置 =====
DATASET_DIR = os.path.expanduser("~/yolo_dataset")
MODEL_SIZE = "n"   # n=nano, s=small, m=medium
EPOCHS = 100
IMG_SIZE = 640

# 1. 生成 data.yaml
yaml_path = os.path.join(DATASET_DIR, "data.yaml")
yaml_content = f"""
path: {DATASET_DIR}
train: images
val: images
nc: 2
names:
  0: fire_extinguisher
  1: fire
"""
with open(yaml_path, "w") as f:
    f.write(yaml_content.strip())
print(f"data.yaml written to {yaml_path}")

# 2. 加载预训练模型开始训练
model = YOLO(f"yolov8{MODEL_SIZE}.pt")

results = model.train(
    data=yaml_path,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    patience=20,        # 20个epoch不提升就早停
    batch=8,
    device="cpu",       # WSL 用 CPU；有 GPU 改 "0"
    verbose=True,
)

# 3. 导出最佳模型
best_pt = os.path.join(os.path.dirname(results.save_dir), "weights", "best.pt")
export_path = os.path.join(DATASET_DIR, "fire_extinguisher_fire.pt")
if os.path.exists(best_pt):
    import shutil
    shutil.copy(best_pt, export_path)
    print(f"\n[OK] Best model exported to: {export_path}")
