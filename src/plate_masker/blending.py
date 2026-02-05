import cv2
import numpy as np


def blur_region(img: np.ndarray, poly: np.ndarray) -> np.ndarray:
    poly = np.array(poly, dtype=np.int32)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)

    blurred = cv2.GaussianBlur(img, (35, 35), 0)

    result = img.copy()
    result[mask == 255] = blurred[mask == 255]
    return result


def prepare_plaque(plaque_img: np.ndarray) -> np.ndarray:
    if plaque_img is None:
        raise ValueError("Не удалось загрузить плашку")

    if plaque_img.shape[2] == 3:
        alpha = np.ones(
            (plaque_img.shape[0], plaque_img.shape[1]), dtype=np.uint8
        ) * 255
        plaque_img = cv2.merge([plaque_img, alpha])

    return plaque_img


def warp_and_blend(
    img: np.ndarray,
    plaque: np.ndarray,
    dst_pts: np.ndarray
) -> np.ndarray:
    h, w = plaque.shape[:2]

    src_pts = np.array(
        [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    H, W = img.shape[:2]

    warped = cv2.warpPerspective(
        plaque,
        M,
        (W, H),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    mask = warped[:, :, 3]
    mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    warped_bgr = warped[:, :, :3]
    mask_norm = (mask / 255.0)[:, :, None]

    result = img * (1 - mask_norm) + warped_bgr * mask_norm
    return result.astype(np.uint8)
