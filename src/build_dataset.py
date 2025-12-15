from pathlib import Path
import random
import shutil
import sys
from typing import List


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

TRAIN_RATIO = 0.85          # train / val
MAX_NEG_RATIO = 2.0         # max negatives per positive in train
SEED = 42


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"

IMAGES_RAW = DATASET_ROOT / "images_raw"
LABELS_SRC = DATASET_ROOT / "labels_cvat" / "labels" / "train"

OUT_IMAGES_TRAIN = DATASET_ROOT / "images" / "train"
OUT_IMAGES_VAL   = DATASET_ROOT / "images" / "val"
OUT_LABELS_TRAIN = DATASET_ROOT / "labels" / "train"
OUT_LABELS_VAL   = DATASET_ROOT / "labels" / "val"


def find_images(images_dir: Path) -> List[Path]:
    return [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]


def make_dirs():
    for p in [
        OUT_IMAGES_TRAIN, OUT_IMAGES_VAL,
        OUT_LABELS_TRAIN, OUT_LABELS_VAL
    ]:
        p.mkdir(parents=True, exist_ok=True)


def split_list(items: List[Path], ratio: float, seed: int):
    if not items:
        return [], []
    rnd = random.Random(seed)
    items = items.copy()
    rnd.shuffle(items)
    cut = int(len(items) * ratio)
    return items[:cut], items[cut:]


def cap_negatives(train_neg: List[Path], num_pos: int, max_ratio: float):
    if num_pos <= 0:
        return []
    limit = int(num_pos * max_ratio)
    return train_neg[:min(limit, len(train_neg))]


def copy_pair(img: Path, img_dst: Path, lbl_dst: Path):
    shutil.copy2(IMAGES_RAW / img.name, img_dst / img.name)

    src_lbl = LABELS_SRC / f"{img.stem}.txt"
    dst_lbl = lbl_dst / f"{img.stem}.txt"

    if src_lbl.exists():
        shutil.copy2(src_lbl, dst_lbl)
    else:
        dst_lbl.write_text("")  # empty label for negative


def write_data_yaml():
    data = f"""path: {DATASET_ROOT.as_posix()}
train: images/train
val: images/val
nc: 1
names: ['license_plate']
"""
    (DATASET_ROOT / "data.yaml").write_text(data)



def main():
    if not IMAGES_RAW.exists():
        print(f"[ERROR] images_raw not found: {IMAGES_RAW}")
        sys.exit(2)

    if not LABELS_SRC.exists():
        print(f"[ERROR] labels_cvat not found: {LABELS_SRC}")
        sys.exit(2)

    make_dirs()
    random.seed(SEED)

    positives = []
    negatives = []

    for img in find_images(IMAGES_RAW):
        lbl = LABELS_SRC / f"{img.stem}.txt"

        if lbl.exists() and lbl.stat().st_size > 0:
            positives.append(img)
        else:
            negatives.append(img)

    print("Dataset discovery:")
    print(f"  Positives: {len(positives)}")
    print(f"  Negatives: {len(negatives)}")

    # split separately
    train_pos, val_pos = split_list(positives, TRAIN_RATIO, SEED)
    train_neg, val_neg = split_list(negatives, TRAIN_RATIO, SEED + 1)

    # cap negatives in train
    train_neg = cap_negatives(train_neg, len(train_pos), MAX_NEG_RATIO)

    train_all = train_pos + train_neg
    val_all = val_pos + val_neg

    random.shuffle(train_all)
    random.shuffle(val_all)

    # copy train
    for img in train_all:
        copy_pair(img, OUT_IMAGES_TRAIN, OUT_LABELS_TRAIN)

    # copy val
    for img in val_all:
        copy_pair(img, OUT_IMAGES_VAL, OUT_LABELS_VAL)

    write_data_yaml()

    print("\nFinal:")
    print(f"  Train: {len(train_all)}  (pos={len(train_pos)}, neg={len(train_neg)})")
    print(f"  Val:   {len(val_all)}    (pos={len(val_pos)}, neg={len(val_neg)})")
    print("DONE.")


if __name__ == "__main__":
    main()
