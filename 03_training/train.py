"""
Stage 3: Training — runs on Colab (T4 GPU), not locally.

Trains YOLOv8 on the lunar tiles produced in Stage 2. Mars tiles are never used here —
they are held out entirely for Stage 4 (zero-shot evaluation).

Setup (run once per Colab session):
    !pip install ultralytics -q

Expected layout inside Colab (after uploading/unzipping the Stage 2 output):
    /content/yolo_dataset/
        dataset.yaml
        images/{train,val}/*.png   <- these are Moon-only splits from Stage 2's spatial split
        labels/{train,val}/*.txt

Only checkpoints are written to Google Drive (not the dataset itself, which stays local
to the Colab session and is re-uploaded each time): two files, overwritten every run,
per the established checkpoint convention — latest.pt and best.pt.
"""
import os
import shutil
from ultralytics import YOLO

DATASET_YAML = "/content/yolo_dataset/dataset.yaml"
DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/GeoAI_Portfolio/Project2_CraterDetection/checkpoints"

MODEL_VARIANT = "yolov8s.pt"   # upgraded from nano: baseline run showed loss still improving at
                                # epoch 100 with capacity likely a limiting factor given ~5,700
                                # training instances; small variant still fits a T4 comfortably.
IMG_SIZE = 640                  # matches Stage 2's tile size exactly, no resizing distortion
EPOCHS = 150                    # baseline hadn't plateaued by epoch 100 (patience=20 never triggered)
BATCH_SIZE = 16                 # lower this if Colab reports CUDA OOM with the larger model
PATIENCE = 25
MAX_DET = 400                   # baseline run had tiles with up to 337 crater instances; the
                                 # ultralytics default (300) silently caps recall on dense tiles.
                                 # Set explicitly with margin rather than relying on auto-correction.

# Augmentation overrides: geometric augmentations (flip, rotate, scale, mosaic) are
# still valid on elevation/slope/hillshade data and help generalization. HSV color-jitter
# augmentations are disabled — they're designed for photographic color variation, which
# is meaningless on channels that aren't color in the first place, and would just inject
# noise into terrain values the model needs to read precisely.
AUGMENTATION_OVERRIDES = dict(
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=15.0,      # small rotations: a crater looks like a crater at any orientation
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,        # combines 4 tiles into one training image — helps with our tile count
    scale=0.2,
)


def train():
    assert os.path.exists(DATASET_YAML), f"Missing: {DATASET_YAML} — upload/unzip Stage 2's output first."

    model = YOLO(MODEL_VARIANT)
    results = model.train(
        data=DATASET_YAML,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        max_det=MAX_DET,
        project="/content/runs",
        name="crater_detection",
        exist_ok=True,
        **AUGMENTATION_OVERRIDES,
    )
    return results


def sync_checkpoints_to_drive():
    run_dir = "/content/runs/crater_detection/weights"
    last_ckpt = f"{run_dir}/last.pt"
    best_ckpt = f"{run_dir}/best.pt"
    assert os.path.exists(last_ckpt) and os.path.exists(best_ckpt), \
        "Checkpoints not found — did training finish at least one epoch?"

    os.makedirs(DRIVE_CHECKPOINT_DIR, exist_ok=True)
    shutil.copy(last_ckpt, f"{DRIVE_CHECKPOINT_DIR}/latest.pt")
    shutil.copy(best_ckpt, f"{DRIVE_CHECKPOINT_DIR}/best.pt")
    print(f"Synced latest.pt and best.pt to {DRIVE_CHECKPOINT_DIR}")


def validate():
    """Quick in-domain (lunar val split) sanity check right after training."""
    best_ckpt = "/content/runs/crater_detection/weights/best.pt"
    model = YOLO(best_ckpt)
    metrics = model.val(data=DATASET_YAML, imgsz=IMG_SIZE, max_det=MAX_DET)
    print(f"Lunar val mAP50: {metrics.box.map50:.4f} | mAP50-95: {metrics.box.map:.4f}")
    return metrics


if __name__ == "__main__":
    train()
    sync_checkpoints_to_drive()
    validate()