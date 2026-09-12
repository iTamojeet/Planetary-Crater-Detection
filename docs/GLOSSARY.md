# Glossary — Project 2: Planetary Crater Detection

Reference for terms used across Stage 1 (data acquisition) and Stage 2 (tiling/labels). Grouped by theme.

## Data sources

- **DEM (Digital Elevation Model)** — a raster where each pixel value is ground elevation, not a photo. What we feed the model, instead of a camera image.
- **LOLA (Lunar Orbiter Laser Altimeter)** — laser altimeter on NASA's LRO spacecraft; source of the lunar DEM. Measures elevation directly by timing a laser pulse's round trip.
- **MOLA (Mars Orbiter Laser Altimeter)** — the Mars equivalent, flown on Mars Global Surveyor.
- **HRSC (High Resolution Stereo Camera)** — a camera on Mars Express; its stereo images are blended with MOLA to fill gaps and sharpen detail, giving the 200 m/px "MOLA+HRSC blend" we used.
- **Robbins Crater Database** — Stuart Robbins' hand-measured global crater catalogs (Moon: 2019, Mars: 2012/2020) — plain tables of `(lat, lon, diameter)` per crater, not images. We convert these rows into bounding boxes ourselves.
- **Astropedia / USGS Astrogeology** — the USGS's planetary-data hosting site, the source for most files here.

## Coordinates & projections

- **Planetocentric coordinates** — lat/lon defined relative to a body's center of mass (as opposed to "planetographic," relative to the local surface normal). The convention these catalogs use.
- **Positive-east longitude, 0–360°** — one of two common longitude conventions; the other is −180° to 180°. The same point can be written either way (e.g., 300° = −60°). Every bug we hit in Stage 1–2 traced back to a file or catalog silently using one convention while our code assumed the other.
- **Equirectangular / Simple Cylindrical projection** — the flattening method these global DEMs use: longitude and latitude map straight onto x/y with no distortion correction, just a linear scale (unlike a Mercator projection). This is why Stage 1's `lonlat → pixel` math could be simple linear interpolation instead of a full reprojection.
- **Geographic CRS vs. Projected CRS** — geographic = coordinates are literally degrees of lat/lon. Projected = coordinates are in real-world linear units (here, meters), the result of applying a projection to a geographic CRS. Our Moon file is projected (meters); our Mars file is geographic (degrees) — hence needing different pixel-conversion code for each in Stage 2.
- **CRS (Coordinate Reference System)** — the full specification of what a file's coordinates mean: the datum (here, a sphere sized to that specific planet, not Earth's WGS84), the projection (or lack of one), and the units.

## File format details

- **Scale/offset (GeoTIFF metadata)** — some GeoTIFFs store compressed integer values plus a `scale`/`offset` to reconstruct the real value (`real = pixel * scale + offset`). LOLA does this (scale=0.5) to store elevation in smaller int16 files instead of float32 — halves file size, but silently gives you wrong values if you forget to apply it.
- **`/vsicurl/`** — GDAL's mechanism for opening a remote file over HTTP as if it were local, fetching only the byte ranges actually read (via HTTP Range requests). Lets you crop a small region out of a multi-GB cloud-hosted file without downloading the whole thing. (We ended up not using this in practice — USGS's server timed out for scripted access — but it's why the windowed-read approach was worth trying first.)

## Geomorphology / terrain analysis

- **Hillshade** — a simulated illumination of terrain from a chosen sun angle (azimuth + altitude), producing shadows that make subtle relief — like degraded crater rims — visible in a way a flat elevation colormap doesn't.
- **Slope** — steepness at each pixel, derived from how fast elevation changes across neighboring pixels (a spatial gradient). Crater rims and walls show up as high-slope pixels.
- **[Elevation, Slope, Hillshade] stack** — our 3-channel "image" fed to YOLOv8 in place of an RGB photo. Each channel captures a different aspect of terrain shape rather than color.

## Dataset construction (ML side)

- **Tiling** — cutting a large raster into small fixed-size chunks (640×640 px here) because object detectors are trained on fixed input sizes, and a whole regional DEM is far too large to feed in one piece.
- **YOLO label format** — one text file per image, one line per object: `class x_center y_center width height`, all four numbers normalized to 0–1 relative to the tile's size. `class` is always `0` here since we only detect one thing: "crater".
- **Bounding box clipping (tile-edge craters)** — a crater whose circle straddles two tiles gets a box drawn in *each* tile it overlaps, clipped to that tile's edges. We drop it entirely if the clipped sliver is too small (<40% of its true width/height) to avoid teaching the model on a near-invisible fragment.
- **Spatial train/val split** — reserving whole geographic columns of tiles for validation (rightmost 20% here) instead of randomly shuffling individual tiles. Prevents "leakage": a randomly-held-out tile sitting right next to a training tile would let the model partly memorize local terrain rather than genuinely generalize.
## Model evaluation

## Model evaluation

- **Zero-shot test set** — the Mars tiles are never touched during training; they only appear at evaluation time, to test whether crater detection learned purely from the Moon transfers to a planet the model has never seen.
- **mAP (mean Average Precision)** — the standard object-detection accuracy metric: for each confidence threshold, compute precision and recall, trace out the precision-recall curve, and take the area under it. `mAP50` uses a lenient overlap threshold (predicted box overlaps ground truth by ≥50% to count as correct); `mAP50-95` averages that score across multiple stricter overlap thresholds (50% to 95%), so it's a harder, more localization-sensitive number.
- **Precision** — of all the boxes the model drew, what fraction were real craters (not false alarms).
- **Recall** — of all the real craters, what fraction the model actually found (not missed).
- **Confidence threshold** — the model outputs a confidence score per candidate box; only boxes above this cutoff are counted as a "detection." Lower it and recall usually rises (more boxes kept) at the cost of precision (more false positives slip through), and vice versa — this is the underlying tradeoff a PR curve visualizes across every possible cutoff at once.
- **Domain gap** — the performance difference between a model's in-domain test set (similar to what it trained on) and an out-of-domain test set (genuinely different data it never saw during training). Here: Moon val performance vs. Mars zero-shot performance. The whole point of Project 2 is measuring this gap, not eliminating it.