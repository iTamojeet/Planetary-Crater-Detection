"""
Stage 4 (advanced): Diagnostic breakdown of the Moon vs. Mars domain gap.

evaluate.py answers "how big is the gap." This answers "what specifically is causing it" —
by stratifying performance by physical crater size (not pixel size — size in km, since Moon
and Mars tiles were cropped at different resolutions), by individual region (not just
planet-level aggregate), and by characterizing false positives directly rather than only
reporting a single precision number.

This does its own IoU-matching from scratch rather than relying on Ultralytics' internal
mAP machinery, specifically so we can slice the results by size/region/confidence in ways
the built-in metrics don't expose.

Requires eval_sets/{moon,mars}/{images,labels} to already exist — run evaluate.py first.

Setup:
    pip install ultralytics numpy matplotlib
"""
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

CHECKPOINT = "checkpoints/best.pt"
EVAL_DIR = "eval_sets"
IMG_SIZE = 640
MAX_DET = 400
OPERATING_CONF = 0.25   # fixed decision threshold, matches the qualitative grid in evaluate.py
IOU_MATCH_THRESH = 0.5  # standard "counts as a correct detection" overlap threshold

# Ground-truth pixel resolution per body from Stage 1/2 — needed because "small" and "large"
# mean different pixel counts on Moon (118 m/px) vs Mars (200 m/px). Binning by physical
# diameter, not pixel size, is what makes the two domains comparable at all.
RESOLUTION_M_PER_PX = {"moon": 118.0, "mars": 200.0}
SIZE_BIN_EDGES_KM = [0, 3, 10, 30, float("inf")]
SIZE_BIN_LABELS = ["<3km", "3-10km", "10-30km", ">30km"]

STEM_PATTERN = re.compile(r"^(moon|mars)_(.+)_r(\d+)_c(\d+)$")


def iou_xyxy(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_stem(stem):
    m = STEM_PATTERN.match(stem)
    assert m, f"Filename doesn't match expected pattern: {stem}"
    body, region, ty, tx = m.groups()
    return body, region


def load_gt_boxes(label_path, img_size=IMG_SIZE):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            if not line.strip():
                continue
            _, xc, yc, bw, bh = map(float, line.split())
            x0, y0 = (xc - bw / 2) * img_size, (yc - bh / 2) * img_size
            x1, y1 = (xc + bw / 2) * img_size, (yc + bh / 2) * img_size
            boxes.append((x0, y0, x1, y1))
    return boxes


def size_bin_km(box, resolution_m):
    w, h = box[2] - box[0], box[3] - box[1]
    diam_km = ((w + h) / 2) * resolution_m / 1000.0
    for i in range(len(SIZE_BIN_EDGES_KM) - 1):
        if SIZE_BIN_EDGES_KM[i] <= diam_km < SIZE_BIN_EDGES_KM[i + 1]:
            return SIZE_BIN_LABELS[i]
    return SIZE_BIN_LABELS[-1]


def match_predictions(pred_boxes, pred_confs, gt_boxes, iou_thresh=IOU_MATCH_THRESH):
    """Greedy matching, highest-confidence prediction first. Returns per-prediction
    (is_tp, matched_iou), the set of matched ground-truth indices, and the count of
    ground-truth boxes left unmatched (false negatives)."""
    order = np.argsort(-pred_confs)
    matched_gt = set()
    tp_flags, tp_ious = [None] * len(pred_boxes), [0.0] * len(pred_boxes)

    for idx in order:
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            i = iou_xyxy(pred_boxes[idx], gt)
            if i > best_iou:
                best_iou, best_j = i, j
        if best_iou >= iou_thresh:
            matched_gt.add(best_j)
            tp_flags[idx], tp_ious[idx] = True, best_iou
        else:
            tp_flags[idx], tp_ious[idx] = False, 0.0

    return tp_flags, tp_ious, matched_gt


def evaluate_domain(model, domain):
    img_paths = sorted(glob.glob(f"{EVAL_DIR}/{domain}/images/*.png"))
    assert img_paths, f"No images found for domain '{domain}' — did evaluate.py run first?"

    records = []      # one row per prediction: domain, region, size_bin, is_tp, iou, confidence
    fn_records = []    # one row per missed ground truth: domain, region, size_bin
    resolution_m = RESOLUTION_M_PER_PX[domain]

    for img_path in img_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        body, region = parse_stem(stem)
        label_path = img_path.replace("/images/", "/labels/").replace(".png", ".txt")

        gt_boxes = load_gt_boxes(label_path)
        result = model.predict(img_path, imgsz=IMG_SIZE, conf=OPERATING_CONF,
                                max_det=MAX_DET, verbose=False)[0]
        pred_boxes = result.boxes.xyxy.cpu().numpy()
        pred_confs = result.boxes.conf.cpu().numpy()

        if len(pred_boxes) == 0:
            for gt in gt_boxes:
                fn_records.append((domain, region, size_bin_km(gt, resolution_m)))
            continue

        tp_flags, tp_ious, matched_gt = match_predictions(pred_boxes, pred_confs, gt_boxes)

        for box, conf, is_tp, iou in zip(pred_boxes, pred_confs, tp_flags, tp_ious):
            records.append((domain, region, size_bin_km(box, resolution_m), is_tp, iou, conf))

        for j, gt in enumerate(gt_boxes):
            if j not in matched_gt:
                fn_records.append((domain, region, size_bin_km(gt, resolution_m)))

    return records, fn_records


def summarize(records, fn_records, group_key_idx, group_labels=None):
    """group_key_idx: 0=domain, 1=region, 2=size_bin. Returns dict[group] -> stats."""
    groups = {}
    for rec in records:
        key = rec[group_key_idx]
        groups.setdefault(key, {"tp": 0, "fp": 0, "fn": 0, "ious": []})
        is_tp, iou = rec[3], rec[4]
        if is_tp:
            groups[key]["tp"] += 1
            groups[key]["ious"].append(iou)
        else:
            groups[key]["fp"] += 1
    for rec in fn_records:
        key = rec[group_key_idx]
        groups.setdefault(key, {"tp": 0, "fp": 0, "fn": 0, "ious": []})
        groups[key]["fn"] += 1

    out = {}
    keys = group_labels if group_labels else sorted(groups.keys())
    for key in keys:
        if key not in groups:
            continue
        g = groups[key]
        tp, fp, fn = g["tp"], g["fp"], g["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        mean_iou = float(np.mean(g["ious"])) if g["ious"] else float("nan")
        out[key] = dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, mean_iou=mean_iou)
    return out


def print_table(title, summary):
    print(f"\n== {title} ==")
    header = f"{'Group':<16}{'TP':>6}{'FP':>6}{'FN':>6}{'Precision':>11}{'Recall':>9}{'MeanIoU':>9}"
    print(header)
    print("-" * len(header))
    for key, s in summary.items():
        print(f"{key:<16}{s['tp']:>6}{s['fp']:>6}{s['fn']:>6}"
              f"{s['precision']:>11.3f}{s['recall']:>9.3f}{s['mean_iou']:>9.3f}")


def plot_recall_by_size(size_summary_by_domain, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(SIZE_BIN_LABELS))
    width = 0.35
    for i, domain in enumerate(["moon", "mars"]):
        recalls = [size_summary_by_domain[domain].get(b, {}).get("recall", 0) for b in SIZE_BIN_LABELS]
        ax.bar(x + i * width, recalls, width, label=domain.capitalize())
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(SIZE_BIN_LABELS)
    ax.set_ylabel("Recall")
    ax.set_title("Recall by physical crater diameter — Moon vs. Mars")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def plot_recall_by_region(region_summary, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    regions = list(region_summary.keys())
    recalls = [region_summary[r]["recall"] for r in regions]
    colors = ["steelblue" if r.startswith(("mare", "highlands_d", "south")) else "firebrick"
              for r in regions]
    ax.bar(regions, recalls, color=colors)
    ax.set_ylabel("Recall")
    ax.set_title("Recall by region (blue=Moon, red=Mars)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def plot_confidence_distributions(records, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, domain in zip(axes, ["moon", "mars"]):
        tp_confs = [r[5] for r in records if r[0] == domain and r[3]]
        fp_confs = [r[5] for r in records if r[0] == domain and not r[3]]
        bins = np.linspace(OPERATING_CONF, 1.0, 20)
        ax.hist(tp_confs, bins=bins, alpha=0.6, label="True positive", color="green")
        ax.hist(fp_confs, bins=bins, alpha=0.6, label="False positive", color="red")
        ax.set_title(domain.capitalize())
        ax.set_xlabel("Confidence")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    fig.suptitle("Prediction confidence: true vs. false positives")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    assert os.path.exists(CHECKPOINT), f"Missing: {CHECKPOINT}"
    for d in ["moon", "mars"]:
        assert os.path.isdir(f"{EVAL_DIR}/{d}/images"), f"Missing {EVAL_DIR}/{d}/images — run evaluate.py first."

    model = YOLO(CHECKPOINT)

    all_records, all_fn = [], []
    for domain in ["moon", "mars"]:
        recs, fns = evaluate_domain(model, domain)
        all_records.extend(recs)
        all_fn.extend(fns)
        print(f"{domain}: {len(recs)} predictions, {len(fns)} missed ground-truth craters")

    domain_summary = summarize(all_records, all_fn, group_key_idx=0, group_labels=["moon", "mars"])
    print_table("Overall (operating point conf=0.25, IoU=0.5)", domain_summary)

    region_summary = summarize(all_records, all_fn, group_key_idx=1)
    print_table("Per-region breakdown", region_summary)

    size_summary_by_domain = {}
    for domain in ["moon", "mars"]:
        recs_d = [r for r in all_records if r[0] == domain]
        fns_d = [f for f in all_fn if f[0] == domain]
        size_summary_by_domain[domain] = summarize(recs_d, fns_d, group_key_idx=2,
                                                     group_labels=SIZE_BIN_LABELS)
        print_table(f"Size-stratified — {domain}", size_summary_by_domain[domain])

    os.makedirs(f"{EVAL_DIR}/advanced", exist_ok=True)
    plot_recall_by_size(size_summary_by_domain, f"{EVAL_DIR}/advanced/recall_by_size.png")
    plot_recall_by_region(region_summary, f"{EVAL_DIR}/advanced/recall_by_region.png")
    plot_confidence_distributions(all_records, f"{EVAL_DIR}/advanced/confidence_distributions.png")

    print(f"\nAll plots saved under {EVAL_DIR}/advanced/")
