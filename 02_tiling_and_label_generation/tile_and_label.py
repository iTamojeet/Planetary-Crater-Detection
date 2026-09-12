"""
Stage 2: Tiling + Label Generation — runs locally, no GPU needed.

Input:  ../01_data_acquisition/dem_crops/{moon,mars}/<region>.tif + <region>_craters.csv
Output: yolo_dataset/images/{train,val}/*.png + yolo_dataset/labels/{train,val}/*.txt

Setup:
    pip install rasterio pandas numpy pillow pyproj
"""
import os
import math
import numpy as np
import pandas as pd
import rasterio
import pyproj
from PIL import Image

INPUT_DIR = "../01_data_acquisition/dem_crops"
OUTPUT_DIR = "yolo_dataset"
TILE_SIZE = 640
VAL_FRACTION = 0.2  # rightmost fraction of tile-columns per region reserved for val (spatial split)
MIN_KEEP_FRACTION = 0.4  # a crater clipped by a tile edge must retain >=40% of its width/height to keep
MIN_BOX_PX = 5  # drop boxes smaller than this in either dimension after clipping

# A crater's true diameter in pixels below which it's not a realistic detection target at this
# resolution and channel stack (hillshade/slope signal from a ~2px-wide feature is mostly noise,
# not shape). Raised from an earlier looser threshold after the Stage 3 baseline showed recall
# capped around 0.3 — a large share of the catalog's smallest craters were being asked of the
# model as labeled targets while being close to physically indistinguishable from noise. This
# does not change what's geologically real, only what we ask the detector to find.
MIN_DIAMETER_PX = 5

LAT_COL, LON_COL, DIAM_COL = "LAT_CIRC_IMG", "LON_CIRC_IMG", "DIAM_CIRC_IMG"

REGIONS = {
    "moon": ["mare_procellarum", "highlands_descartes", "south_pole"],
    "mars": ["highlands_sabaea", "plains_amazonis", "argyre_rim"],
}


def compute_hillshade(elevation, cellsize_m, azimuth=315, altitude=45):
    """Standard hillshade from an elevation array. Crater rims show up via cast shadow —
    a flat colormap on raw elevation alone washes out small/degraded craters."""
    az_rad = np.radians(360.0 - azimuth + 90)
    alt_rad = np.radians(altitude)
    dy, dx = np.gradient(elevation, cellsize_m)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    shaded = (np.sin(alt_rad) * np.sin(slope) +
              np.cos(alt_rad) * np.cos(slope) * np.cos(az_rad - aspect))
    hillshade = np.clip(shaded * 255, 0, 255)
    slope_deg = np.degrees(np.pi / 2.0 - slope)
    return hillshade, slope_deg


def normalize_to_uint8(arr, pct_clip=2):
    lo, hi = np.percentile(arr, [pct_clip, 100 - pct_clip])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def build_3channel_image(elevation, cellsize_m):
    hillshade, slope_deg = compute_hillshade(elevation, cellsize_m)
    return np.dstack([
        normalize_to_uint8(elevation),
        normalize_to_uint8(slope_deg),
        normalize_to_uint8(hillshade),
    ])  # H x W x 3, channels = [elevation, slope, hillshade]


def get_cellsize_m_and_lonlat_to_xy(ds):
    """Returns (cellsize_in_meters, fn(lon, lat) -> (x, y in ds's own CRS units)).
    Moon crops are in a projected CRS (meters); Mars crops here are in a geographic
    CRS (degrees) — handled generically via pyproj rather than assuming either."""
    crs = pyproj.CRS.from_wkt(ds.crs.to_wkt())
    cellsize_native = abs(ds.transform.a)

    if crs.is_geographic:
        radius_m = crs.ellipsoid.semi_major_metre
        cellsize_m = cellsize_native * (math.pi / 180.0) * radius_m
        left = ds.bounds.left  # detect this file's native lon domain, same as Stage 1

        def lonlat_to_xy(lon, lat):
            if left < -1 and lon > 180:
                lon = lon - 360
            elif left >= -1 and lon < 0:
                lon = lon + 360
            return lon, lat
    else:
        cellsize_m = cellsize_native  # projected equirectangular here is already in meters
        geodetic = crs.geodetic_crs
        transformer = pyproj.Transformer.from_crs(geodetic, crs, always_xy=True)

        def lonlat_to_xy(lon, lat):
            return transformer.transform(lon, lat)

    return cellsize_m, lonlat_to_xy


def tile_region(body, region, split_counts):
    tif_path = f"{INPUT_DIR}/{body}/{region}.tif"
    csv_path = f"{INPUT_DIR}/{body}/{region}_craters.csv"
    assert os.path.exists(tif_path), f"Missing: {tif_path} (run Stage 1 first)"
    assert os.path.exists(csv_path), f"Missing: {csv_path} (run Stage 1 first)"

    with rasterio.open(tif_path) as ds:
        elevation = ds.read(1)
        transform = ds.transform
        cellsize_m, lonlat_to_xy = get_cellsize_m_and_lonlat_to_xy(ds)

    craters = pd.read_csv(csv_path)
    img = build_3channel_image(elevation, cellsize_m)
    h, w = elevation.shape
    n_tiles_y, n_tiles_x = h // TILE_SIZE, w // TILE_SIZE
    val_start_col = int(n_tiles_x * (1 - VAL_FRACTION))

    # Bucket: (tile_row, tile_col) -> list of YOLO label lines
    tile_labels = {(ty, tx): [] for ty in range(n_tiles_y) for tx in range(n_tiles_x)}
    n_dropped_small = 0

    for _, crater in craters.iterrows():
        lon, lat, diam_km = crater[LON_COL], crater[LAT_COL], crater[DIAM_COL]
        x, y = lonlat_to_xy(lon, lat)
        row, col = rasterio.transform.rowcol(transform, x, y)
        radius_px = (diam_km * 1000.0 / 2.0) / cellsize_m
        if radius_px * 2 < MIN_DIAMETER_PX:
            n_dropped_small += 1
            continue  # below the realistic detection floor at this resolution — see MIN_DIAMETER_PX

        x_min, x_max = col - radius_px, col + radius_px
        y_min, y_max = row - radius_px, row + radius_px
        orig_w, orig_h = x_max - x_min, y_max - y_min

        tx_min, tx_max = int(x_min // TILE_SIZE), int(x_max // TILE_SIZE)
        ty_min, ty_max = int(y_min // TILE_SIZE), int(y_max // TILE_SIZE)

        for ty in range(max(ty_min, 0), min(ty_max, n_tiles_y - 1) + 1):
            for tx in range(max(tx_min, 0), min(tx_max, n_tiles_x - 1) + 1):
                tile_x0, tile_y0 = tx * TILE_SIZE, ty * TILE_SIZE
                clip_x_min = max(x_min, tile_x0) - tile_x0
                clip_x_max = min(x_max, tile_x0 + TILE_SIZE) - tile_x0
                clip_y_min = max(y_min, tile_y0) - tile_y0
                clip_y_max = min(y_max, tile_y0 + TILE_SIZE) - tile_y0
                clip_w, clip_h = clip_x_max - clip_x_min, clip_y_max - clip_y_min

                if clip_w < MIN_KEEP_FRACTION * orig_w or clip_h < MIN_KEEP_FRACTION * orig_h:
                    continue  # mostly clipped off by the tile edge — drop rather than mislabel
                if clip_w < MIN_BOX_PX or clip_h < MIN_BOX_PX:
                    continue

                xc = (clip_x_min + clip_x_max) / 2 / TILE_SIZE
                yc = (clip_y_min + clip_y_max) / 2 / TILE_SIZE
                bw, bh = clip_w / TILE_SIZE, clip_h / TILE_SIZE
                tile_labels[(ty, tx)].append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    for (ty, tx), labels in tile_labels.items():
        split = "val" if tx >= val_start_col else "train"
        tile_img = img[ty * TILE_SIZE:(ty + 1) * TILE_SIZE, tx * TILE_SIZE:(tx + 1) * TILE_SIZE]
        stem = f"{body}_{region}_r{ty}_c{tx}"

        img_dir = f"{OUTPUT_DIR}/images/{split}"
        lbl_dir = f"{OUTPUT_DIR}/labels/{split}"
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        Image.fromarray(tile_img).save(f"{img_dir}/{stem}.png")
        with open(f"{lbl_dir}/{stem}.txt", "w") as f:
            f.write("\n".join(labels))

        split_counts[split]["tiles"] += 1
        split_counts[split]["boxes"] += len(labels)

    print(f"{body}/{region}: {n_tiles_y}x{n_tiles_x} tiles, "
          f"{sum(len(v) for v in tile_labels.values())} boxes total "
          f"({n_dropped_small} craters dropped as below {MIN_DIAMETER_PX}px diameter)")


def write_dataset_yaml():
    with open(f"{OUTPUT_DIR}/dataset.yaml", "w") as f:
        f.write(f"""path: {os.path.abspath(OUTPUT_DIR)}
train: images/train
val: images/val
nc: 1
names: ['crater']
""")


if __name__ == "__main__":
    print("tile_and_label.py v3 (minimum detectable-diameter filter)")
    split_counts = {"train": {"tiles": 0, "boxes": 0}, "val": {"tiles": 0, "boxes": 0}}
    for body, regions in REGIONS.items():
        for region in regions:
            tile_region(body, region, split_counts)

    write_dataset_yaml()
    print("\n== Summary ==")
    for split, counts in split_counts.items():
        print(f"{split}: {counts['tiles']} tiles, {counts['boxes']} crater boxes")
    print(f"\nWritten to {OUTPUT_DIR}/ — zip this folder and upload to Colab for Stage 3 (training).")