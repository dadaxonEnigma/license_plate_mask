import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import torch


# =====================
# CONFIG
# =====================
MODEL_PATH = "../notebooks/runs/segment/train/weights/best.pt"
IMAGES_DIR = Path("../test_images")

CONF_THRES = 0.5
IMGSZ = 1024


# =====================
# MAIN
# =====================
def main():
    model = YOLO(MODEL_PATH)

    images = [p for p in IMAGES_DIR.iterdir()
              if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".webp"]]

    if not images:
        print("❌ Нет изображений")
        return

    total = len(images)
    with_det = 0
    no_det = 0

    confs = []
    areas = []

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        r = model.predict(
            img,
            conf=CONF_THRES,
            imgsz=IMGSZ,
            verbose=False,
            device=0 if torch.cuda.is_available() else "cpu"
        )[0]

        if r.masks is None or len(r.masks.xy) == 0:
            no_det += 1
            continue

        with_det += 1

        if r.boxes is not None and r.boxes.conf is not None:
            confs.extend(r.boxes.conf.cpu().numpy().tolist())

        if r.boxes is not None:
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box
                areas.append((x2 - x1) * (y2 - y1))

    print("\n===== DETECTION RATIO =====")
    print(f"Total images: {total}")
    print(f"With detection: {with_det}")
    print(f"No detection: {no_det}")
    print(f"Detection rate: {with_det / total:.2%}")

    if confs:
        print("\n===== CONFIDENCE =====")
        print(f"Mean: {np.mean(confs):.3f}")
        print(f"Median: {np.median(confs):.3f}")
        print(f"Min / Max: {min(confs):.3f} / {max(confs):.3f}")

    if areas:
        print("\n===== AREA (px²) =====")
        print(f"Mean: {np.mean(areas):.0f}")
        print(f"Min / Max: {min(areas):.0f} / {max(areas):.0f}")


if __name__ == "__main__":
    main()
