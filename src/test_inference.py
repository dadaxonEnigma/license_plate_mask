import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import torch


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)

    if pts.shape[0] > 4:
        hull = cv2.convexHull(pts).reshape(-1, 2)
        if hull.shape[0] == 4:
            pts = hull

    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect


def blur_region(img, poly):
    poly = np.array(poly, dtype=np.int32)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)

    blurred = cv2.GaussianBlur(img, (35, 35), 0)

    result = img.copy()
    result[mask == 255] = blurred[mask == 255]
    return result


def prepare_plaque(plaque_img):
    if plaque_img is None:
        raise ValueError("Не удалось загрузить плашку.")

    if plaque_img.shape[2] == 3:
        alpha = np.ones((plaque_img.shape[0], plaque_img.shape[1]), dtype=np.uint8) * 255
        plaque_img = cv2.merge([plaque_img, alpha])

    return plaque_img


def warp_and_blend(img, plaque, dst_pts):
    h, w = plaque.shape[:2]

    src_pts = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    H, W = img.shape[:2]

    warped = cv2.warpPerspective(
        plaque, M, (W, H),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    mask = warped[:, :, 3]
    mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    warped_bgr = warped[:, :, :3]
    mask_norm = (mask / 255.0)[:, :, np.newaxis]

    result = img * (1 - mask_norm) + warped_bgr * mask_norm
    return result.astype(np.uint8)


def is_bad_quad(quad, min_side=8):
    d01 = np.linalg.norm(quad[0] - quad[1])
    d12 = np.linalg.norm(quad[1] - quad[2])
    d23 = np.linalg.norm(quad[2] - quad[3])
    d30 = np.linalg.norm(quad[3] - quad[0])

    if min(d01, d12, d23, d30) < min_side:
        return True

    return False


def mask_all_plates(image_path, model, plaque, out_path, conf=0.8):

    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка загрузки изображения: {image_path}")
        return False  

    result_img = img.copy()

    pred = model.predict(
        image_path,
        imgsz=1024,
        conf=conf,
        verbose=False,
        device=0 if torch.cuda.is_available() else "cpu"
    )[0]

    if pred.masks is None or len(pred.masks.xy) == 0:
        print("Номера не найдены.")
        cv2.imwrite(out_path, result_img)
        return False

    masks = pred.masks.xy
    found_any = False

    for i, poly in enumerate(masks):
        found_any = True
        try:
            poly = np.array(poly, dtype=np.float32)

            quad = order_points(poly)

            if is_bad_quad(quad):
                print(f'Маска {i}: плохая геометрия — blur')
                result_img = blur_region(result_img, poly)
                continue
            try:
                result_img = warp_and_blend(result_img, plaque, quad)
            except Exception:
                print(f'Плашка не смогла встать, применяем blur')
                result_img = blur_region(result_img, poly)

        except Exception as e:
            print(f"Ошибка при обработке объекта {i}: {e}")
            continue

    cv2.imwrite(out_path, result_img)
    print(f"Сохранено: {out_path}")
    return found_any


def process_folder(folder_path, model_path, plaque_path, output_dir):

    model = YOLO(model_path)
    plaque = prepare_plaque(cv2.imread(plaque_path, cv2.IMREAD_UNCHANGED))

    folder = Path(folder_path)
    output_dir = Path(output_dir)

    processed_dir = output_dir / "processed"
    no_number_dir = output_dir / "no_number"

    processed_dir.mkdir(parents=True, exist_ok=True)
    no_number_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in folder.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".webp"]]

    print(f"Найдено файлов: {len(images)}")

    for idx, img_path in enumerate(images, start=1):
        print(f"[{idx}/{len(images)}] {img_path.name}")

        temp_out = output_dir / f"{img_path.stem}_masked_temp.jpg"

        found = mask_all_plates(
            image_path=str(img_path),
            model=model,
            plaque=plaque,
            out_path=str(temp_out)
        )

        if found:
            final_path = processed_dir / f"{img_path.stem}_processed.jpg"
        else:
            final_path = no_number_dir / f"{img_path.stem}_no_number.jpg"

        temp_out.rename(final_path)

    print("Готово.")


if __name__ == "__main__":
    MODEL_PATH = "../notebooks/runs/segment/train/weights/best (4).pt"
    PLAQUE_PATH = "../assets/overlay.png"

    process_folder(
        folder_path="../test_images/new_cars",
        model_path=MODEL_PATH,
        plaque_path=PLAQUE_PATH,
        output_dir="../outputs2"
    )
