# Stage 3: Train the lunar crater detector

This stage fine-tunes a pretrained YOLOv8 nano detector to find craters in lunar
terrain tiles. It runs in Google Colab with a T4 GPU. The model is trained and
validated on **Moon-only** data; Mars tiles are deliberately excluded so that
Stage 4 can measure zero-shot cross-planet generalization.

## Change log (v2)

The first baseline run (`yolov8n`, 100 epochs) reached mAP50 = 0.311, mAP50-95 = 0.128,
precision 0.49, recall 0.32. Loss curves were healthy and still improving at epoch 100
(patience never triggered), so this was not a training failure — the likely limiting factors
were model capacity and label difficulty, not the optimization process. This version addresses
both:

- **Stage 2 (`tile_and_label.py`) now filters craters below `MIN_DIAMETER_PX = 5`** before they
  become labels. A large share of the Robbins catalog's smallest craters are only a few pixels
  wide at this resolution — close to indistinguishable from noise in the hillshade/slope channels
  regardless of how well the model trains. Asking the detector to find those as if they were
  reliable targets both caps achievable recall and adds label noise to legitimate mid-size craters
  near tile boundaries. Re-run Stage 2 and re-upload `yolo_dataset.zip` before training with this
  version — box counts will be lower than the original run's.
- **Base model upgraded to `yolov8s.pt`** (was `yolov8n.pt`). With roughly 5,700 training
  instances and loss still trending down at epoch 100, the nano variant's ~3M parameters were
  plausibly a capacity bottleneck; small (~11M parameters) still fits a free-tier T4 comfortably.
- **`max_det` set explicitly to 400.** The baseline run logged tiles with up to 337 crater
  instances, above Ultralytics' default cap of 300 — silently capping recall on the densest
  tiles until Ultralytics' own warning caught it. Set with margin here instead of relying on
  the automatic correction.
- **Epochs raised to 150, patience to 25**, since the baseline hadn't plateaued by 100.

## What this stage consumes and produces

**Input:** the YOLO-format dataset generated in Stage 2. Every PNG is a
640 × 640, three-channel `uint8` terrain image with channels in this order:

1. locally percentile-normalized elevation;
2. locally percentile-normalized slope in degrees; and
3. locally percentile-normalized hillshade (azimuth 315°, altitude 45°).

Labels are one-class YOLO bounding boxes: `0 x_center y_center width height`,
with normalized coordinates. Class `0` is `crater`.

**Output:** the run directory (`results.csv`, plots, confusion matrix, and
weights), plus two Drive checkpoints:

| File | Meaning |
| --- | --- |
| `latest.pt` | Weights from the last completed epoch; useful for resuming/debugging. |
| `best.pt` | Weights from the epoch with the best validation fitness; use this for evaluation. |

## Important dataset contract

`train.py` assumes Moon-only training. The current Stage 2 generator writes both
Moon and Mars tiles, so do **not** upload its output unchanged. Remove files
whose names begin with `mars_` from both the image and matching label folders
before training. If Mars appears in either `train` or `val`, the zero-shot Mars
evaluation is invalid.

The spatial validation split is created per region in Stage 2: the rightmost
20% of tile columns are validation, rather than a random-tile split. This helps
avoid overly optimistic metrics from adjacent, highly similar tiles appearing
in both splits.

The dataset YAML must describe the directory used in Colab, not the absolute
path recorded on the machine that generated the tiles:

```yaml
path: /content/yolo_dataset
train: images/train
val: images/val
nc: 1
names: [crater]
```

## Colab setup and run

1. In Colab, select **Runtime → Change runtime type → T4 GPU**, then mount
   Google Drive.

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !nvidia-smi
   !pip install -q ultralytics
   ```

2. Upload `yolo_dataset.zip` from Stage 2 to Colab and extract it. The folder
   must end up at `/content/yolo_dataset`.

   ```python
   !unzip -q /content/yolo_dataset.zip -d /content
   !find /content/yolo_dataset -type f -name 'mars_*' -delete
   ```

   If the archive extracts into an additional directory level, move or rename
   that extracted directory so that `/content/yolo_dataset/images/train` exists.

3. Replace `dataset.yaml` with the Colab-safe version above, then check that no
   Mars files remain and that every image has a corresponding label file.

   ```python
   %%writefile /content/yolo_dataset/dataset.yaml
   path: /content/yolo_dataset
   train: images/train
   val: images/val
   nc: 1
   names: [crater]
   ```

   ```python
   !find /content/yolo_dataset -type f -name 'mars_*'
   !find /content/yolo_dataset/images/train -type f -name 'moon_*.png' | wc -l
   !find /content/yolo_dataset/images/val -type f -name 'moon_*.png' | wc -l
   ```

   The first command must print no paths. With the dataset currently in this
   repository, the expected Moon-only counts are 132 training tiles and 48
   validation tiles.

4. Upload `train.py` to `/content` (or clone this repository) and run it.

   ```python
   !python /content/train.py
   ```

`train.py` trains, copies weights to Drive, and then evaluates `best.pt` on the
lunar validation split. The Drive destination is configured by
`DRIVE_CHECKPOINT_DIR`; change that constant if your Drive layout differs.

## Training configuration

| Setting | Value | Rationale |
| --- | ---: | --- |
| Base weights | `yolov8s.pt` | Small variant; upgraded from nano after the baseline run showed capacity was plausibly limiting. Still fits a free-tier T4. |
| Image size | 640 | Matches the Stage 2 tile size; no resizing distortion. |
| Epochs | 150 | Baseline run hadn't plateaued by epoch 100. |
| Batch size | 16 | Lower it if Colab runs out of memory with the larger model. |
| Early stopping patience | 25 epochs | Stops when validation fitness no longer improves. |
| Max detections per image | 400 | Baseline run had tiles with up to 337 instances; set above that with margin rather than the Ultralytics default of 300. |
| Classes | 1 (`crater`) | Crater detection only. |
| Rotation | ±15° | Crater orientation has no semantic meaning. |
| Horizontal/vertical flip | 0.5 / 0.5 | Terrain orientation is not a label feature. |
| Scale | 0.2 | Adds modest size variation. |
| Mosaic | 1.0 | Increases scene composition variety for the small tile set. |
| HSV augmentation | disabled | The inputs are terrain measurements, not color photographs. |

All other optimizer, learning-rate, loss, data-loader, and validation options
are the defaults of the installed Ultralytics version. Record that version in
each experiment because defaults may change:

```python
import ultralytics, torch
print('ultralytics:', ultralytics.__version__)
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
```

## Results and acceptance checks

The script prints lunar validation `mAP50` and `mAP50-95`. Inspect the artifacts
under `/content/runs/crater_detection/`, especially `results.csv`, the PR curve,
the confusion matrix, and qualitative validation predictions. Prefer `best.pt`
for all reported results.

Before treating a run as usable, confirm:

- `best.pt` and `latest.pt` exist both in the run directory and Drive.
- The validation set contains only `moon_` tiles.
- Curves are stable and validation loss does not diverge from training loss.
- False positives and missed craters are visually plausible rather than caused
  by corrupt channels or label alignment errors.

Do not tune on Mars tiles or use Mars metrics to select this checkpoint. Mars
belongs only to the held-out cross-domain evaluation stage.

## Why YOLOv8

YOLOv8 is a single-stage, anchor-free detector with multi-scale features (P3,
P4, and P5). Its PAN-FPN neck mixes spatial detail with semantic context across
scales, which suits craters ranging from a few pixels to hundreds of pixels.
The three terrain layers occupy the model's ordinary three-channel input slot;
no architectural change is needed. YOLOv8's pretrained weights provide a useful
initialization, while fine-tuning adapts the detector to terrain morphology.