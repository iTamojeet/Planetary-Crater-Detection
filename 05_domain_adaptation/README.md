# Stage 5: Domain Adaptation (final experiment stage)

Tests the two directions raised by Stage 4's findings, without retraining on Mars data —
retraining would defeat the point of a zero-shot test, so both experiments only touch
inference-time inputs (resampling, threshold choice), never model weights.

## Setup

```
pip install ultralytics numpy matplotlib pillow
```

Requires `../04_evaluation/eval_sets/{moon,mars}/{images,labels}` and
`../04_evaluation/checkpoints/best.pt` to already exist (Stage 4 output).

## Run

```
cd 05_domain_adaptation
python3 domain_adapt.py
```

CPU inference at the resampled 1088px size is noticeably slower than the native 640px path —
expect several minutes rather than seconds.

## Experiment A: Resolution matching

**Hypothesis (from Stage 4):** Mars's small-crater recall collapse is a resolution artifact —
a crater spans fewer pixels at Mars's 200 m/px than the same physical crater would at the
Moon's 118 m/px, so bicubic-upsampling Mars tiles by the resolution ratio (~1.695x, to 1088px)
before inference should recover much of that gap.

**Result: mostly did not work.**

| Size bin | Native recall | Resampled recall | Native precision | Resampled precision |
|---|---:|---:|---:|---:|
| <3km | 0.085 | 0.139 | 0.141 | 0.078 |
| 3-10km | 0.682 | 0.665 | 0.263 | 0.222 |
| 10-30km | 0.692 | 0.606 | 0.577 | 0.594 |
| >30km | 0.534 | 0.442 | 0.530 | 0.361 |

The smallest bin's recall did rise, consistent with the hypothesis — but its precision fell
even further (false positives nearly tripled, 1799 → 5675), and every other size bin got worse
on both metrics. Bicubic upsampling doesn't add real information; it interpolates existing
pixels onto a larger canvas. That modestly helps small craters look more like what the model
expects in scale, but simultaneously blurs and distorts craters that were already reasonably
scaled, and amplifies interpolation artifacts into new spurious detections. Net effect across
the whole tile: a bad trade.

This refines rather than overturns Stage 4's finding: resolution is clearly *a* contributing
factor to the small-crater recall gap, but naive resampling is not a working fix for it. A real
fix would likely need either genuine super-resolution (recovering plausible high-frequency
detail, not just interpolating) or training-time exposure to multiple resolutions — both
meaningfully larger undertakings than an inference-time transform.

## Experiment B: Per-domain threshold calibration

**Hypothesis (from Stage 4):** Mars precision collapses at a shared confidence threshold
because Mars false positives cluster at low-to-moderate confidence in a way the Moon's don't —
so Mars likely has its own, higher, better operating threshold.

**Result: confirmed, and it's a real, free improvement.**

| Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0.25 (shared, used throughout Stage 4) | 0.242 | 0.252 | 0.247 |
| 0.45 (Mars-specific best-F1) | 0.548 | 0.208 | 0.302 |

Moving Mars's decision threshold from the shared 0.25 to its own best-F1 point at 0.45 more
than doubles precision (0.242 → 0.548) for a modest recall cost (0.252 → 0.208), a ~22%
improvement in F1 — with no retraining, no architecture change, and no new data. This is the
one intervention in this stage that earns a genuine deployment recommendation: if this detector
were ever run on Mars data in practice, it should not use the same confidence threshold tuned
on Moon data.

## Overall verdict

Of the two evidence-backed directions from Stage 4, only one — threshold calibration — produced
an unambiguous improvement. The other — resolution matching — surfaced a more nuanced picture:
the underlying hypothesis (resolution drives the small-crater gap) still looks directionally
correct, but the simplest possible fix for it doesn't work cleanly, and actually made most of
the detector's behavior worse. That asymmetry — a cheap software fix works, a more physically
motivated fix doesn't — is itself a legitimate and honest finding to report, rather than
something to paper over by only including the intervention that worked.

## Output

```
adaptation_results/
├── resampled_tiles/                    <- Mars tiles upsampled to ~118m/px equivalent
├── resolution_matching_effect.png      <- native vs. resampled recall by size bin
└── mars_threshold_sweep.png            <- precision/recall/F1 vs. confidence threshold
```

## This closes the experimental portion of Project 2

Summary chain across all four evaluation-facing stages:
- **Stage 3:** in-domain lunar baseline, mAP50 = 0.362.
- **Stage 4:** zero-shot Mars, mAP50 = 0.211 (41.7% relative drop) — driven by a resolution-linked
  small-crater recall collapse and a Mars-specific confidence-calibration failure, not a
  shape-recognition failure (crater detection itself transfers reasonably well once craters are
  large enough to be well-resolved).
- **Stage 5:** confirmed threshold calibration as a real, free fix; confirmed resolution matching
  as a more complex problem than a single inference-time transform can solve.