import numpy as np
import cv2


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32)

    if pts.shape[0] > 4:
        hull = cv2.convexHull(pts).reshape(-1, 2)
        if hull.shape[0] == 4:
            pts = hull

    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]   
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def is_bad_quad(quad: np.ndarray, min_side: int = 8) -> bool:
    d01 = np.linalg.norm(quad[0] - quad[1])
    d12 = np.linalg.norm(quad[1] - quad[2])
    d23 = np.linalg.norm(quad[2] - quad[3])
    d30 = np.linalg.norm(quad[3] - quad[0])

    return min(d01, d12, d23, d30) < min_side
