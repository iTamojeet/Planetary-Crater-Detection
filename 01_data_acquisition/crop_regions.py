"""
Stage 1: Data Acquisition — run locally on Mac (no Colab needed for this part).
Only windowed reads happen here; outputs are small crops safe to upload to Colab later for training.

Setup:
    pip install rasterio pandas
    (If rasterio fails to build: brew install gdal, then retry)

Place these files in ./data/ before running (DEMs + crater catalogs already unzipped/extracted by you):
    data/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif
    data/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif
    data/lunar_craters.csv    <- extracted from lunar_crater_database_robbins_2018.zip
    data/mars_craters.csv     <- extracted from Catalog_Mars_Release_2020_1kmPlus_FullMorphData.csv.zip
                                 (actually tab-separated; script auto-detects the delimiter)
"""
import os
import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import from_bounds

DATA_DIR = "data"
LOLA_PATH = f"{DATA_DIR}/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"
MOLA_HRSC_PATH = f"{DATA_DIR}/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif"

LUNAR_REGIONS = {
    "mare_procellarum": (300, 320, 10, 25),      # mare, sparse large craters
    "highlands_descartes": (10, 25, -12, 3),     # dense, overlapping, degraded craters
    "south_pole": (0, 40, -85, -70),             # extreme lighting
}
MARS_REGIONS = {
    "highlands_sabaea": (40, 60, 0, 15),         # ancient, densest crater terrain
    "plains_amazonis": (195, 215, -5, 10),       # young smooth volcanic plains
    "argyre_rim": (310, 330, -55, -40),          # complex/degraded large craters
}

LUNAR_CRATERS_FILE = f"{DATA_DIR}/lunar_crater_database_robbins_2018_bundle/data/lunar_crater_database_robbins_2018.csv"
MARS_CRATERS_FILE = f"{DATA_DIR}/Catalog_Mars_Release_2020_1kmPlus_FullMorphData.csv"


def read_region(local_path, lon_min, lon_max, lat_min, lat_max):
    with rasterio.open(local_path) as ds:
        left, bottom, right, top = ds.bounds

        # Auto-detect whether the file uses 0-360 or -180/180 longitude convention,
        # since USGS mosaics aren't consistent about this despite similar metadata wording.
        if left < -1:
            lon_domain = (-180, 180)
            def to_domain(lon):
                return lon - 360 if lon > 180 else lon
        else:
            lon_domain = (0, 360)
            def to_domain(lon):
                return lon + 360 if lon < 0 else lon

        def lon_to_x(lon):
            lon = to_domain(lon)
            frac = (lon - lon_domain[0]) / (lon_domain[1] - lon_domain[0])
            return left + frac * (right - left)

        def lat_to_y(lat):
            # lat 90 -> top (max y), lat -90 -> bottom (min y); monotonic increasing with lat.
            frac = (lat - (-90)) / (90 - (-90))
            return bottom + frac * (top - bottom)

        x_min, x_max = sorted([lon_to_x(lon_min), lon_to_x(lon_max)])
        y_min, y_max = sorted([lat_to_y(lat_min), lat_to_y(lat_max)])
        print(f"  ds.bounds={ds.bounds} | window x=[{x_min:.1f},{x_max:.1f}] y=[{y_min:.1f},{y_max:.1f}]")
        window = from_bounds(x_min, y_min, x_max, y_max, transform=ds.transform)
        data = ds.read(1, window=window)
        scale = ds.scales[0] if ds.scales and ds.scales[0] else 1.0
        offset = ds.offsets[0] if ds.offsets and ds.offsets[0] else 0.0
        elevation_m = data.astype(np.float32) * scale + offset
        return elevation_m, ds.window_transform(window), ds.crs


def save_crop(elevation_m, transform, crs, out_path):
    with rasterio.open(out_path, "w", driver="GTiff", height=elevation_m.shape[0], width=elevation_m.shape[1],
                        count=1, dtype="float32", transform=transform, crs=crs) as dst:
        dst.write(elevation_m, 1)


def crop_dems():
    os.makedirs("dem_crops/moon", exist_ok=True)
    os.makedirs("dem_crops/mars", exist_ok=True)

    for name, bbox in LUNAR_REGIONS.items():
        elev, tfm, crs = read_region(LOLA_PATH, *bbox)
        print(name, "->", elev.shape, f"elev range [{elev.min():.0f}, {elev.max():.0f}] m")
        save_crop(elev, tfm, crs, f"dem_crops/moon/{name}.tif")

    for name, bbox in MARS_REGIONS.items():
        elev, tfm, crs = read_region(MOLA_HRSC_PATH, *bbox)
        print(name, "->", elev.shape, f"elev range [{elev.min():.0f}, {elev.max():.0f}] m")
        save_crop(elev, tfm, crs, f"dem_crops/mars/{name}.tif")


def load_craters_file(path):
    """Loads an already-extracted crater catalog file. Auto-detects comma vs tab
    delimiter (Robbins Mars releases are tab-separated despite the .csv name)."""
    assert os.path.exists(path), f"Missing: {path} — extract the catalog zip yourself and place it here."
    df = pd.read_csv(path, sep=None, engine="python")
    print(path, "->", df.shape)
    print(df.columns.tolist())
    return df


def fetch_lunar_craters():
    return load_craters_file(LUNAR_CRATERS_FILE)


def fetch_mars_craters_df():
    return load_craters_file(MARS_CRATERS_FILE)


def slice_craters(df, regions, out_dir, lat_col, lon_col):
    for name, (lon_min, lon_max, lat_min, lat_max) in regions.items():
        subset = df[df[lon_col].between(lon_min, lon_max) & df[lat_col].between(lat_min, lat_max)]
        subset.to_csv(f"{out_dir}/{name}_craters.csv", index=False)
        print(name, "->", len(subset), "craters")


if __name__ == "__main__":
    print("crop_regions.py v4 (correct Mars column names)")
    for p in [LOLA_PATH, MOLA_HRSC_PATH]:
        assert os.path.exists(p), f"Missing: {p}"

    print("== Cropping DEM regions ==")
    crop_dems()

    print("\n== Lunar crater catalog ==")
    moon_craters = fetch_lunar_craters()
    input("^ Confirm lat/lon column names, then edit MOON_LAT_COL/MOON_LON_COL below if needed. Press Enter...")
    MOON_LAT_COL, MOON_LON_COL = "LAT_CIRC_IMG", "LON_CIRC_IMG"
    slice_craters(moon_craters, LUNAR_REGIONS, "dem_crops/moon", MOON_LAT_COL, MOON_LON_COL)

    print("\n== Mars crater catalog ==")
    mars_craters = fetch_mars_craters_df()
    input("^ Confirm lat/lon column names, then edit MARS_LAT_COL/MARS_LON_COL below if needed. Press Enter...")
    MARS_LAT_COL, MARS_LON_COL = "LAT_CIRC_IMG", "LON_CIRC_IMG"  # confirmed from printed columns above
    slice_craters(mars_craters, MARS_REGIONS, "dem_crops/mars", MARS_LAT_COL, MARS_LON_COL)

    print("\nDone. Output in ./dem_crops/ — safe to zip and upload to Colab for Stage 2+.")