# Stage 4: Zero-Shot Evaluation (Moon vs. Mars)

This is the project's core result: how much detection performance degrades when a YOLOv8
detector trained entirely on lunar terrain is applied, unmodified, to Martian terrain it has
never seen. Two scripts, two levels of resolution:

- `evaluate.py` — the headline comparison: aggregate metrics, PR curves, a qualitative grid.
- `advanced_eval.py` — a from-scratch IoU-matching diagnostic that breaks the result down by
  physical crater size, by individual region, and by prediction confidence, to isolate *why*
  the gap exists rather than just stating that it does.

## Setup

```
pip install ultralytics matplotlib pillow numpy
```

1. Download `best.pt` from Google Drive (`GeoAI_Portfolio/Project2_CraterDetection/checkpoints/best.pt`)
   and place it at `04_evaluation/checkpoints/best.pt`.
2. Confirm `../02_tiling_and_label_generation/yolo_dataset/` still has both Moon and Mars tiles. The copy
   uploaded to Colab for training had Mars tiles deleted deliberately — this stage needs the
   original local copy with everything intact.

## Run

```
cd 04_evaluation
python3 evaluate.py          # aggregate metrics, PR curves, qualitative grid
python3 advanced_eval.py     # size/region/confidence breakdown (run evaluate.py first)
```

## Method

- **Moon set:** the exact validation split from Stage 2/3 — never trained on, but drawn from
  the same three regions (and general terrain distribution) as training.
- **Mars set:** every Mars tile pooled from both the original `train` and `val` folders, since
  none of them were used in training regardless of which column they originally landed in.
  This uses the full held-out Mars data rather than an arbitrary subset.
- `evaluate.py` reports Ultralytics' own `mAP50`/`mAP50-95` (computed by an internal confidence
  sweep, not a fixed threshold) alongside a qualitative prediction grid.
- `advanced_eval.py` fixes a single shared operating threshold (confidence ≥ 0.25, matching the
  qualitative grid) and does its own greedy IoU-matching (IoU ≥ 0.5) so results can be sliced by
  physical crater diameter — converted to km using each body's actual ground resolution (118 m/px
  Moon, 200 m/px Mars), not by pixel size, since a fixed pixel threshold means different physical
  sizes on the two bodies — and by individual region rather than only a two-way aggregate.

A methodological note worth keeping visible: the first version of `evaluate.py` passed the fixed
confidence threshold into Ultralytics' `val()` call, which computes mAP via its own internal
sweep starting near confidence zero — forcing a high threshold there truncates that sweep and
silently deflates every reported number. Both scripts as delivered here compute `val()` metrics
unthresholded, and only apply a fixed threshold in the manual-matching diagnostic where an
explicit operating point is actually the right thing to fix.

## Results

### Headline comparison (`evaluate.py`, unthresholded mAP)

| Metric | Moon | Mars | Relative drop |
|---|---:|---:|---:|
| mAP50 | 0.362 | 0.211 | 41.7% |
| mAP50-95 | 0.159 | 0.090 | 43.3% |
| Precision | 0.530 | 0.549 | −3.6% (no real change) |
| Recall | 0.355 | 0.210 | 40.9% |

At Ultralytics' own best-F1 operating point per domain, precision looks essentially flat and the
entire gap looks like a recall story. That turns out to be an artifact of each domain being
scored at its own separately-chosen best threshold — see below.

### Fixed shared threshold (`advanced_eval.py`, conf=0.25, IoU=0.5 for both domains)

| Domain | TP | FP | FN | Precision | Recall | Mean IoU |
|---|---:|---:|---:|---:|---:|---:|
| Moon | 1286 | 587 | 3364 | 0.687 | 0.277 | 0.725 |
| Mars | 1217 | 3820 | 3609 | 0.242 | 0.252 | 0.673 |

Once both domains are forced through the *same* threshold — the only fair comparison, and the
only realistic one for actual deployment, since you can't know a planet-specific optimal
threshold in advance — Mars precision collapses to 0.242 against the Moon's 0.687. The
"flat precision" result above was real but threshold-dependent, not domain-independent.

### Recall by physical crater diameter

| Diameter | Moon recall | Mars recall |
|---|---:|---:|
| <3 km | 0.248 | 0.085 |
| 3–10 km | 0.525 | 0.682 |
| 10–30 km | 0.320 | 0.692 |
| >30 km | 0.200 | 0.534 |

### Recall by region

| Region | Body | Recall |
|---|---|---:|
| mare_procellarum | Moon | 0.630 |
| highlands_descartes | Moon | 0.337 |
| south_pole | Moon | 0.159 |
| highlands_sabaea | Mars | 0.292 |
| plains_amazonis | Mars | 0.212 |
| argyre_rim | Mars | 0.140 |

(Precision per region, from the same run: Mars regions sit at 0.19–0.26 across all three;
`mare_procellarum` is the best performer on both metrics of any region on either body.)

## Findings

**1. Crater shape generalizes across planets; small-object recall does not — and the reason is
resolution, not geology.** Once a crater is large enough to be well-resolved at Mars's coarser
200 m/px scale, Mars recall matches or exceeds the Moon's at every size bin above 3 km. The
catastrophic recall loss lives almost entirely in the smallest bin (<3 km: 0.248 vs. 0.085). A
3 km crater occupies roughly half as many pixels on Mars as the same physical crater would at
lunar resolution — this is consistent with a sampling/resolution effect, not a failure of the
model to recognize Martian crater rims as craters. Stage 1's deliberate choice of a 200 m/px
Mars product (over the native 463 m/px MOLA-only DEM) to narrow this exact gap was directionally
correct, but a 118 vs. 200 m/px difference is still large enough to dominate the small-object
regime.

**2. The model's confidence scores don't mean the same thing on Mars as on the Moon.** At a
shared operating threshold, Mars is flooded with false positives clustered at the low-confidence
end of the range (visible directly in `confidence_distributions.png`) in a way the Moon simply
isn't. This is a calibration failure, not a shape-recognition failure: the network is producing
plausible-looking confidence scores for spurious detections on terrain textures it never saw
during lunar-only training, and doing so more often at low-to-moderate confidence than the Moon's
false positives do. Practically, this means precision at *any* fixed threshold will look
substantially worse on Mars even in size ranges where recall is strong.

**3. The precision problem is planet-wide, not one bad region.** All three Mars regions land in
the same 0.19–0.26 precision band regardless of terrain type (ancient highlands, young plains, or
a degraded basin rim), which argues against "one unusual region skewed the average" and for a
genuine property of the Moon-to-Mars transfer itself.

**4. Qualitatively, isolated well-formed craters transfer almost perfectly.** In
`domain_gap_visual.png`, sparse tiles on both bodies (`mare_procellarum` on the Moon,
`plains_amazonis_r2_c5` on Mars) show near-pixel-perfect box alignment on large, clean craters.
The same figure also shows the Mars-specific false-positive mode directly: boxes drawn on plain
terrain texture with no corresponding ground-truth crater nearby, on tiles with no dense crater
field to explain them.

## Limitations

- The `>30 km` size bin has very few samples on both bodies (single digits to low tens); treat
  that row as suggestive, not conclusive.
- `advanced_eval.py`'s greedy IoU-matching at a single fixed threshold is a diagnostic tool, not
  a drop-in replacement for Ultralytics' mAP, which integrates over many thresholds. The two
  scripts are answering related but distinct questions and their precision/recall numbers should
  not be expected to match exactly.
- Region-level sample sizes vary substantially (e.g. `argyre_rim` has far fewer craters than
  `highlands_sabaea`), so per-region numbers carry different amounts of statistical weight.

## Output

```
eval_sets/
├── moon/{images,labels}/, mars/{images,labels}/     <- copied/pooled evaluation tiles
├── moon.yaml, mars.yaml                              <- per-domain Ultralytics dataset configs
├── moon_val/, mars_val/                              <- Ultralytics val() output, incl. PR_curve.png
├── domain_gap_visual.png                             <- qualitative Moon vs. Mars grid
└── advanced/
    ├── recall_by_size.png
    ├── recall_by_region.png
    └── confidence_distributions.png
```

## Next

Two specific, evidence-backed directions for Stage 6 (optional domain adaptation), rather than
generic "improve the model":

- **Address the calibration gap, not the shape gap.** Since crater shape itself appears to
  transfer reasonably well, the higher-leverage fix is likely a Mars-specific confidence
  recalibration or a per-domain decision threshold, rather than architectural changes aimed at
  crater recognition itself.
- **Address the resolution mismatch directly**, e.g. by upsampling Mars tiles toward lunar pixel
  scale (or downsampling lunar training tiles toward Mars scale) before inference, to test
  whether the small-crater recall gap is actually eliminated by resolution-matching alone — which
  would be strong confirmatory evidence for Finding 1 above.

Retraining on Mars data is deliberately out of scope for either direction — it would defeat the
purpose of a zero-shot test, which is the entire point of this project.
