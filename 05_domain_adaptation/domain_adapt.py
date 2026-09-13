"""
Stage 5: Domain Adaptation — final stage. Tests the two evidence-backed directions raised in
the Stage 4 findings, without retraining on Mars data (which would defeat the point of a
zero-shot test):

  (A) Resolution matching: Stage 4 found small-crater recall on Mars collapses specifically
      below ~3km, consistent with Mars's coarser 200 m/px scale rather than a shape-recognition
      failure. This resamples Mars tiles up to Moon's effective 118 m/px scale before inference
      and re-measures recall by size bin — if the hypothesis is right, the small-crater gap
      should substantially close.

  (B) Per-domain threshold calibration: Stage 4 found Mars precision collapses at a shared
      confidence threshold because Mars false positives cluster at low-to-moderate confidence
      in a way Moon's don't. This sweeps thresholds on Mars alone (using predictions already
      run once, not re-run per threshold) to find Mars's own best operating point, and reports
      what a domain-aware deployment would actually use.

Reuses matching/scoring logic from Stage 4's advanced_eval.py rather than duplicating it —
run this from within 05_domain_adaptation/ with 04_evaluation/ as a sibling folder.

Requires eval_sets/{moon,mars}/{images,labels} from Stage 4 to already exist.

Setup:
    pip install ultralytics numpy matplotlib pillow
"""
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath("../04_evaluation"))
from advanced_eval import (  # noqa: E402
    load_gt_boxes, size_bin_km, match_predictions, summarize, print_table,
    parse_stem, RESOLUTION_M_PER_PX, SIZE_BIN_LABELS, IMG_SIZE, MAX_DET, IOU_MATCH_THRESH,
    OPERATING_CONF, EVAL_DIR as SOURCE_EVAL_DIR,
)

CHECKPOINT = "../04_evaluation/checkpoints/best.pt"
OUT_DIR = "adaptation_results"

# Ratio needed to make a Mars pixel represent the same ground distance as a Moon pixel.
# A crater that spans N pixels at 200 m/px would span N * RESAMPLE_FACTOR pixels at 118 m/px —
# upsampling Mars tiles by this factor before inference makes crater pixel-footprints match
# what the model actually learned to recognize, rather than changing any real information.
RESAMPLE_FACTOR = RESOLUTION_M_PER_PX["mars"] / RESOLUTION_M_PER_PX["moon"]
RESAMPLED_SIZE = int(round(IMG_SIZE * RESAMPLE_FACTOR / 32) * 32)  # round to a stride-32 multiple


def run_inference_pool(model, img_paths, imgsz, conf_floor=0.05):
    """Runs inference once per image at a low confidence floor, returning raw candidate boxes
    per image so downstream code can filter by any higher threshold without re-running the model."""
    pool = []
    for img_path in img_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        body, region = parse_stem(stem)
        label_path = img_path.replace("/images/", "/labels/").replace(".png", ".txt")
        gt_boxes_native = load_gt_boxes(label_path, img_size=IMG_SIZE)

        result = model.predict(img_path, imgsz=imgsz, conf=conf_floor, max_det=MAX_DET, verbose=False)[0]
        pred_boxes = result.boxes.xyxy.cpu().numpy()
        pred_confs = result.boxes.conf.cpu().numpy()
        pool.append(dict(region=region, gt_boxes=gt_boxes_native, pred_boxes=pred_boxes, pred_confs=pred_confs))
    return pool


def score_pool(pool, conf_threshold, resolution_m, gt_scale=1.0):
    """Filters a pre-computed prediction pool to a given confidence threshold, matches against
    ground truth (scaled by gt_scale if predictions were made on a resized image), and returns
    (records, fn_records) in the same shape advanced_eval.summarize() expects."""
    records, fn_records = [], []
    for entry in pool:
        region = entry["region"]
        gt_boxes = [tuple(c * gt_scale for c in box) for box in entry["gt_boxes"]]
        keep = entry["pred_confs"] >= conf_threshold
        pred_boxes = entry["pred_boxes"][keep]
        pred_confs = entry["pred_confs"][keep]

        if len(pred_boxes) == 0:
            for gt in gt_boxes:
                fn_records.append(("mars", region, size_bin_km(gt, resolution_m)))
            continue

        tp_flags, tp_ious, matched_gt = match_predictions(pred_boxes, pred_confs, gt_boxes, IOU_MATCH_THRESH)
        for box, conf, is_tp, iou in zip(pred_boxes, pred_confs, tp_flags, tp_ious):
            records.append(("mars", region, size_bin_km(box, resolution_m), is_tp, iou, conf))
        for j, gt in enumerate(gt_boxes):
            if j not in matched_gt:
                fn_records.append(("mars", region, size_bin_km(gt, resolution_m)))
    return records, fn_records


def experiment_a_resolution_matching(model):
    """Compares Mars recall-by-size at native resolution vs. resampled to Moon's effective
    resolution, at the same fixed operating threshold used throughout Stage 4."""
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Resolution matching")
    print("=" * 70)

    mars_img_paths = sorted(glob.glob(f"../04_evaluation/{SOURCE_EVAL_DIR}/mars/images/*.png"))
    assert mars_img_paths, "No Mars evaluation images found — run Stage 4's evaluate.py first."

    # Native resolution (baseline, matches Stage 4's advanced_eval.py numbers)
    native_pool = run_inference_pool(model, mars_img_paths, imgsz=IMG_SIZE)
    native_records, native_fn = score_pool(native_pool, OPERATING_CONF, RESOLUTION_M_PER_PX["mars"])
    native_summary = summarize(native_records, native_fn, group_key_idx=2, group_labels=SIZE_BIN_LABELS)
    print_table("Mars @ native 200m/px (baseline)", native_summary)

    # Resampled: upsample each tile, predict at the larger size, scale GT boxes to match.
    os.makedirs(f"{OUT_DIR}/resampled_tiles", exist_ok=True)
    resampled_paths = []
    for img_path in mars_img_paths:
        stem = os.path.basename(img_path)
        img = Image.open(img_path)
        resized = img.resize((RESAMPLED_SIZE, RESAMPLED_SIZE), Image.BICUBIC)
        out_path = f"{OUT_DIR}/resampled_tiles/{stem}"
        resized.save(out_path)
        resampled_paths.append(out_path)

    resampled_pool_raw = run_inference_pool(model, resampled_paths, imgsz=RESAMPLED_SIZE)
    # Re-attach original gt/region info keyed by matching filename stem (resampled dir mirrors names).
    for entry, orig_path in zip(resampled_pool_raw, mars_img_paths):
        stem = os.path.splitext(os.path.basename(orig_path))[0]
        label_path = orig_path.replace("/images/", "/labels/").replace(".png", ".txt")
        entry["gt_boxes"] = load_gt_boxes(label_path, img_size=IMG_SIZE)

    gt_scale = RESAMPLED_SIZE / IMG_SIZE
    resampled_records, resampled_fn = score_pool(
        resampled_pool_raw, OPERATING_CONF, RESOLUTION_M_PER_PX["moon"], gt_scale=gt_scale
    )
    resampled_summary = summarize(resampled_records, resampled_fn, group_key_idx=2, group_labels=SIZE_BIN_LABELS)
    print_table(f"Mars resampled to {RESAMPLED_SIZE}px (effective ~118m/px)", resampled_summary)

    plot_before_after(native_summary, resampled_summary, f"{OUT_DIR}/resolution_matching_effect.png")
    return native_summary, resampled_summary


def plot_before_after(native_summary, resampled_summary, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(SIZE_BIN_LABELS))
    width = 0.35
    native_recalls = [native_summary.get(b, {}).get("recall", 0) for b in SIZE_BIN_LABELS]
    resampled_recalls = [resampled_summary.get(b, {}).get("recall", 0) for b in SIZE_BIN_LABELS]
    ax.bar(x - width / 2, native_recalls, width, label="Mars, native 200m/px")
    ax.bar(x + width / 2, resampled_recalls, width, label="Mars, resampled to ~118m/px")
    ax.set_xticks(x)
    ax.set_xticklabels(SIZE_BIN_LABELS)
    ax.set_ylabel("Recall")
    ax.set_title("Effect of resolution matching on Mars recall, by crater size")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def experiment_b_threshold_calibration(model):
    """Sweeps confidence threshold on Mars alone (single inference pass, many threshold cuts)
    to find Mars's own best-F1 operating point, contrasted with the shared 0.25 used elsewhere."""
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Per-domain threshold calibration")
    print("=" * 70)

    mars_img_paths = sorted(glob.glob(f"../04_evaluation/{SOURCE_EVAL_DIR}/mars/images/*.png"))
    pool = run_inference_pool(model, mars_img_paths, imgsz=IMG_SIZE, conf_floor=0.05)

    thresholds = np.arange(0.10, 0.90, 0.05)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        records, fn_records = score_pool(pool, t, RESOLUTION_M_PER_PX["mars"])
        tp = sum(1 for r in records if r[3])
        fp = sum(1 for r in records if not r[3])
        fn = len(fn_records)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    best_idx = int(np.argmax(f1s))
    best_t = thresholds[best_idx]
    print(f"{'Threshold':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")
    for t, p, r, f1 in zip(thresholds, precisions, recalls, f1s):
        marker = "  <-- best F1" if abs(t - best_t) < 1e-9 else ""
        print(f"{t:>10.2f}{p:>11.3f}{r:>9.3f}{f1:>8.3f}{marker}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, marker="o", label="Precision")
    ax.plot(thresholds, recalls, marker="o", label="Recall")
    ax.plot(thresholds, f1s, marker="o", label="F1")
    ax.axvline(OPERATING_CONF, color="gray", linestyle="--", label=f"Shared threshold ({OPERATING_CONF})")
    ax.axvline(best_t, color="black", linestyle=":", label=f"Mars best-F1 ({best_t:.2f})")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Score")
    ax.set_title("Mars: precision/recall/F1 vs. confidence threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/mars_threshold_sweep.png", dpi=150)
    print(f"Saved {OUT_DIR}/mars_threshold_sweep.png")
    print(f"\nRecommended Mars-specific operating threshold: {best_t:.2f} "
          f"(precision {precisions[best_idx]:.3f}, recall {recalls[best_idx]:.3f})")
    return best_t, precisions[best_idx], recalls[best_idx]


if __name__ == "__main__":
    assert os.path.exists(CHECKPOINT), f"Missing: {CHECKPOINT}"
    os.makedirs(OUT_DIR, exist_ok=True)

    model = YOLO(CHECKPOINT)

    native_summary, resampled_summary = experiment_a_resolution_matching(model)
    best_t, best_p, best_r = experiment_b_threshold_calibration(model)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Resolution matching (<3km bin): "
          f"{native_summary['<3km']['recall']:.3f} -> {resampled_summary['<3km']['recall']:.3f} recall")
    print(f"Threshold calibration: shared conf={OPERATING_CONF} vs. Mars-optimal conf={best_t:.2f} "
          f"(precision {best_p:.3f}, recall {best_r:.3f} at the Mars-specific threshold)")
    print(f"\nAll outputs under {OUT_DIR}/")