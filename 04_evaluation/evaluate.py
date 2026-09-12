"""
Stage 4: Zero-Shot Evaluation — runs locally, CPU inference is fine at this scale.

Compares the Stage 3 checkpoint (trained on Moon only) against:
  (a) the Moon validation split (in-domain — the model has seen similar terrain, just not
      these exact tiles), and
  (b) ALL Mars tiles (zero-shot — none of these were used or seen in training at all).

This is the actual scientific result of the project: the gap between (a) and (b) IS the
domain-gap measurement, not a side detail.

Setup:
    pip install ultralytics matplotlib pillow

Before running:
    1. Download best.pt from Google Drive
       (GeoAI_Portfolio/Project2_CraterDetection/checkpoints/best.pt)
       and place it at ./checkpoints/best.pt
    2. Make sure ../02_tiling_and_label_generation/yolo_dataset/ still has its original Moon+Mars
       content (i.e. you're pointing at the pre-Colab-upload local copy, not the
       Mars-stripped copy that went to Colab).
"""
import os
import glob
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics import YOLO

CHECKPOINT = "checkpoints/best.pt"
SOURCE_DATASET = "../02_tiling_and_label_generation/yolo_dataset"
EVAL_DIR = "eval_sets"
IMG_SIZE = 640
CONF_THRESHOLD = 0.25  # only used for the qualitative prediction grid (a real decision threshold
                        # for "what counts as a detection to draw"). NOT passed to val() — mAP is
                        # computed by sweeping confidence internally at a near-zero threshold, and
                        # forcing a high conf there truncates the sweep and deflates every metric.
MAX_DET = 400           # matches Stage 3's training config


def build_mars_only_set():
    """Mars tiles were never trained on regardless of which Stage 2 split (train/val) they
    originally landed in — so for zero-shot evaluation we pool ALL of them together, not just
    the ones that happened to fall in the 'val' column split."""
    mars_img_dir = f"{EVAL_DIR}/mars/images"
    mars_lbl_dir = f"{EVAL_DIR}/mars/labels"
    os.makedirs(mars_img_dir, exist_ok=True)
    os.makedirs(mars_lbl_dir, exist_ok=True)

    count = 0
    for split in ["train", "val"]:
        for img_path in glob.glob(f"{SOURCE_DATASET}/images/{split}/mars_*.png"):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = f"{SOURCE_DATASET}/labels/{split}/{stem}.txt"
            shutil.copy(img_path, f"{mars_img_dir}/{stem}.png")
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, f"{mars_lbl_dir}/{stem}.txt")
            else:
                open(f"{mars_lbl_dir}/{stem}.txt", "w").close()  # empty label = background tile
            count += 1

    assert count > 0, (
        f"Found 0 Mars tiles under {os.path.abspath(SOURCE_DATASET)}/images/{{train,val}}/mars_*.png\n"
        "This means the source dataset is missing or empty, not a bug in this script.\n"
        "Check: (1) does that folder still exist locally, or was it deleted after zipping for "
        "Colab? (2) is SOURCE_DATASET pointing at the right relative path from wherever you're "
        "running this script? Run: ls ../02_tiling_and_label_generation/yolo_dataset/images/train | head"
    )
    print(f"Pooled {count} Mars tiles (zero-shot test set, spans both original train/val columns)")
    return count


def write_eval_yaml(name, images_dir):
    yaml_path = f"{EVAL_DIR}/{name}.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"""path: {os.path.abspath(EVAL_DIR)}
train: {name}/images
val: {name}/images
nc: 1
names: ['crater']
""")
    return yaml_path


def link_moon_val_set():
    """Reuses Stage 2's existing Moon val split as-is — this is the in-domain comparison arm."""
    moon_img_dir = f"{EVAL_DIR}/moon/images"
    moon_lbl_dir = f"{EVAL_DIR}/moon/labels"
    os.makedirs(moon_img_dir, exist_ok=True)
    os.makedirs(moon_lbl_dir, exist_ok=True)

    count = 0
    for img_path in glob.glob(f"{SOURCE_DATASET}/images/val/moon_*.png"):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = f"{SOURCE_DATASET}/labels/val/{stem}.txt"
        shutil.copy(img_path, f"{moon_img_dir}/{stem}.png")
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, f"{moon_lbl_dir}/{stem}.txt")
        else:
            open(f"{moon_lbl_dir}/{stem}.txt", "w").close()
        count += 1

    assert count > 0, (
        f"Found 0 Moon tiles under {os.path.abspath(SOURCE_DATASET)}/images/val/moon_*.png\n"
        "Same root cause as the Mars check below — verify the source dataset path/existence first."
    )
    print(f"Linked {count} Moon validation tiles (in-domain comparison set)")
    return count


def run_eval(model, name, images_dir):
    yaml_path = write_eval_yaml(name, images_dir)
    metrics = model.val(data=yaml_path, imgsz=IMG_SIZE, max_det=MAX_DET,
                         split="val", project=EVAL_DIR, name=f"{name}_val", exist_ok=True)
    return {
        "domain": name,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
    }


def print_comparison(moon_metrics, mars_metrics):
    print("\n== Domain gap: Moon (in-domain) vs Mars (zero-shot) ==")
    header = f"{'Metric':<12}{'Moon':>10}{'Mars':>10}{'Relative drop':>16}"
    print(header)
    print("-" * len(header))
    for key in ["mAP50", "mAP50-95", "precision", "recall"]:
        moon_v, mars_v = moon_metrics[key], mars_metrics[key]
        drop_pct = (1 - mars_v / moon_v) * 100 if moon_v > 0 else float("nan")
        print(f"{key:<12}{moon_v:>10.4f}{mars_v:>10.4f}{drop_pct:>15.1f}%")


def render_prediction_grid(model, out_path, n_per_domain=3, seed=0):
    """The 'visual asset': elevation-channel heatmap with ground-truth (green) and
    predicted (red) bounding boxes overlaid, for a handful of tiles from each domain."""
    random.seed(seed)
    domains = {
        "Moon (in-domain)": glob.glob(f"{EVAL_DIR}/moon/images/*.png"),
        "Mars (zero-shot)": glob.glob(f"{EVAL_DIR}/mars/images/*.png"),
    }

    fig, axes = plt.subplots(2, n_per_domain, figsize=(5 * n_per_domain, 10))
    for row, (domain_name, img_paths) in enumerate(domains.items()):
        sample = random.sample(img_paths, min(n_per_domain, len(img_paths)))
        for col, img_path in enumerate(sample):
            ax = axes[row][col]
            stem = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = img_path.replace("/images/", "/labels/").replace(".png", ".txt")

            img = np.array(Image.open(img_path))
            elevation_channel = img[:, :, 0]  # channel 0 = elevation, per Stage 2's channel order
            ax.imshow(elevation_channel, cmap="terrain")
            ax.set_title(f"{domain_name}\n{stem}", fontsize=9)
            ax.axis("off")

            h, w = elevation_channel.shape
            if os.path.exists(lbl_path):
                with open(lbl_path) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        _, xc, yc, bw, bh = map(float, line.split())
                        x0, y0 = (xc - bw / 2) * w, (yc - bh / 2) * h
                        ax.add_patch(patches.Rectangle((x0, y0), bw * w, bh * h,
                                                        edgecolor="lime", facecolor="none", linewidth=1))

            result = model.predict(img_path, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
            for box in result.boxes.xyxy.cpu().numpy():
                x0, y0, x1, y1 = box
                ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                                edgecolor="red", facecolor="none", linewidth=1))

    fig.suptitle("Green = ground truth  |  Red = model prediction", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visual comparison grid to {out_path}")


if __name__ == "__main__":
    assert os.path.exists(CHECKPOINT), f"Missing: {CHECKPOINT} — download best.pt from Drive first."
    assert os.path.isdir(SOURCE_DATASET), (
        f"Missing: {os.path.abspath(SOURCE_DATASET)}\n"
        "Your local yolo_dataset/ from Stage 2 isn't where this script expects it. Either it was "
        "deleted after zipping for Colab upload, or you're running this script from somewhere "
        "other than the 04_evaluation/ folder. Fix SOURCE_DATASET at the top of this file, or "
        "re-run Stage 2's tile_and_label.py if the folder is genuinely gone."
    )
    os.makedirs(EVAL_DIR, exist_ok=True)

    link_moon_val_set()
    build_mars_only_set()

    model = YOLO(CHECKPOINT)
    moon_metrics = run_eval(model, "moon", f"{EVAL_DIR}/moon/images")
    mars_metrics = run_eval(model, "mars", f"{EVAL_DIR}/mars/images")

    print_comparison(moon_metrics, mars_metrics)
    render_prediction_grid(model, f"{EVAL_DIR}/domain_gap_visual.png")

    print(f"\nPR curves saved under {EVAL_DIR}/moon_val/ and {EVAL_DIR}/mars_val/ "
          f"(PR_curve.png in each) for direct side-by-side comparison.")