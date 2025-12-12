#!/usr/bin/env python3
"""
prepare_dataset.py

Production-ready dataset preparation for YOLO (segmentation) from a single images folder
and CVAT-style YOLO txt labels.

Features:
- Detects positive (non-empty label), negative (no label), excluded (filename hints masked)
- Splits train/val per-class (positive/negative) with configurable ratios
- Caps negatives in train by max_neg_ratio * num_positive_train to avoid class imbalance
- Creates empty label files for negatives (without overwriting existing non-empty labels)
- Writes data.yaml
- Robust handling of image extensions
- Logging and checks

Usage:
    python prepare_dataset.py \
        --dataset-root dataset \
        --images-raw dataset/images_raw \
        --labels-cvat dataset/labels_cvat/labels/train \
        --train-ratio 0.85 \
        --val-ratio 0.15 \
        --max-neg-ratio 1.5 \
        --seed 42
"""

import argparse
from pathlib import Path
import random
import shutil
import sys
from typing import List, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def find_images(images_dir: Path) -> List[Path]:
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()])

def is_masked_by_name(p: Path, keywords=("masked", "blur", "замаск", "mask")) -> bool:
    n = p.name.lower()
    return any(kw in n for kw in keywords)

def label_path_for_image(labels_dir: Path, img_path: Path) -> Path:
    return labels_dir / (img_path.stem + ".txt")

def has_nonempty_label(lbl_path: Path) -> bool:
    return lbl_path.exists() and lbl_path.stat().st_size > 0

def make_dirs(paths: List[Path]):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def split_list(items: List, ratio: float, seed: int) -> Tuple[List, List]:
    """Split items into train/val using ratio for train portion (e.g., 0.85)."""
    if not items:
        return [], []
    rnd = random.Random(seed)
    items_copy = items.copy()
    rnd.shuffle(items_copy)
    cut = int(len(items_copy) * ratio)
    return items_copy[:cut], items_copy[cut:]

def cap_negatives(train_neg: List[Path], num_pos_train: int, max_neg_ratio: float) -> List[Path]:
    if num_pos_train <= 0:
        # If no positives in train, keep up to 1000 negatives (safety)
        limit = min(len(train_neg), 1000)
    else:
        limit = int(num_pos_train * max_neg_ratio)
        limit = max(1, limit)
        limit = min(limit, len(train_neg))
    return train_neg[:limit]

def copy_and_prepare(files: List[Path], src_images_dir: Path, src_labels_dir: Path, dst_images_dir: Path, dst_labels_dir: Path):
    for img_path in files:
        src_img = src_images_dir / img_path.name
        dst_img = dst_images_dir / img_path.name
        shutil.copy2(src_img, dst_img)

        lbl_src = src_labels_dir / (img_path.stem + ".txt")
        lbl_dst = dst_labels_dir / (img_path.stem + ".txt")

        if lbl_src.exists():
            # Copy existing label (could be empty or non-empty)
            shutil.copy2(lbl_src, lbl_dst)
        else:
            # Create empty label for negative sample (only if it doesn't already exist)
            if not lbl_dst.exists():
                lbl_dst.write_text("")  # create empty file

def write_data_yaml(dataset_root: Path, nc: int = 1, names=None):
    if names is None:
        names = ["license_plate"]
    data_yaml = f"""path: {dataset_root.as_posix()}
train: images/train
val: images/val
nc: {nc}
names: {names}
"""
    (dataset_root / "data.yaml").write_text(data_yaml)

def main(args):
    dataset_root = Path(args.dataset_root).resolve()
    src_images_dir = Path(args.images_raw).resolve()
    src_labels_dir = Path(args.labels_cvat).resolve()

    if not src_images_dir.exists():
        print(f"[ERROR] images_raw directory not found: {src_images_dir}", file=sys.stderr); sys.exit(2)
    if not src_labels_dir.exists():
        print(f"[ERROR] labels_cvat directory not found: {src_labels_dir}", file=sys.stderr); sys.exit(2)

    out_images_train = dataset_root / "images" / "train"
    out_images_val = dataset_root / "images" / "val"
    out_labels_train = dataset_root / "labels" / "train"
    out_labels_val = dataset_root / "labels" / "val"

    make_dirs([out_images_train, out_images_val, out_labels_train, out_labels_val])

    all_images = find_images(src_images_dir)
    if not all_images:
        print("[ERROR] No images found in images_raw. Supported extensions: " + ", ".join(sorted(IMG_EXTS)), file=sys.stderr)
        sys.exit(2)

    positives, negatives, excluded = [], [], []
    for p in all_images:
        if is_masked_by_name(p, tuple(args.mask_keywords)):
            excluded.append(p)
            continue
        lbl = label_path_for_image(src_labels_dir, p)
        if has_nonempty_label(lbl):
            positives.append(p)
        else:
            negatives.append(p)

    print("Dataset discovery:")
    print(f"  Total images found: {len(all_images)}")
    print(f"  Positives (non-empty label): {len(positives)}")
    print(f"  Negatives (no label / empty): {len(negatives)}")
    print(f"  Excluded by filename (masked heuristics): {len(excluded)}")

    # Split positives and negatives separately
    train_pos, val_pos = split_list(positives, args.train_ratio, args.seed)
    train_neg, val_neg = split_list(negatives, args.train_ratio, args.seed + 1)

    # Cap negative in train to avoid imbalance
    train_neg = cap_negatives(train_neg, len(train_pos), args.max_neg_ratio)

    train_final = train_pos + train_neg
    val_final = val_pos + val_neg

    random.seed(args.seed + 2)
    random.shuffle(train_final)
    random.shuffle(val_final)

    print("\nFinal selection counts:")
    print(f"  Train: {len(train_final)} (pos={len(train_pos)}, neg={len(train_neg)})")
    print(f"  Val:   {len(val_final)} (pos={len(val_pos)}, neg={len(val_neg)})")

    # Sanity checks / warnings
    if len(train_pos) == 0:
        print("[WARNING] No positive examples in train set. Model may not learn to detect plates.")
    if len(val_pos) == 0:
        print("[WARNING] No positive examples in val set. Validation won't measure plate detection properly.")
    if len(train_neg) == 0:
        print("[WARNING] No negative examples in train set. Model may produce many false positives.")

    # Copy files and labels
    copy_and_prepare(train_final, src_images_dir, src_labels_dir, out_images_train, out_labels_train)
    copy_and_prepare(val_final, src_images_dir, src_labels_dir, out_images_val, out_labels_val)

    # Write data.yaml
    write_data_yaml(dataset_root, nc=1, names=["license_plate"])

    print("\nFiles copied and data.yaml written.")
    print(f"  images/train: {out_images_train}  (count: {len(list(out_images_train.iterdir()))})")
    print(f"  labels/train: {out_labels_train}  (count: {len(list(out_labels_train.iterdir()))})")
    print(f"  images/val:   {out_images_val}  (count: {len(list(out_images_val.iterdir()))})")
    print(f"  labels/val:   {out_labels_val}  (count: {len(list(out_labels_val.iterdir()))})")
    print(f"\nExcluded sample examples (first 10): {[p.name for p in excluded[:10]]}")
    print("\nDONE.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset (seg) from single images folder + CVAT labels.")
    parser.add_argument("--dataset-root", default="dataset", help="Output dataset root (default: dataset)")
    parser.add_argument("--images-raw", default="dataset/images_raw", help="Source images folder")
    parser.add_argument("--labels-cvat", default="dataset/labels_cvat/labels/train", help="Source labels folder")
    parser.add_argument("--train-ratio", type=float, default=0.85, help="Train ratio per-class (default 0.85)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val ratio per-class (unused if train_ratio provided)")
    parser.add_argument("--max-neg-ratio", type=float, default=1.5, help="Max negative per positive in train (default 1.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mask-keywords", nargs="*", default=["masked", "mask", "blur", "замаск"], help="Filename keywords to detect pre-masked images")
    args = parser.parse_args()
    main(args)
