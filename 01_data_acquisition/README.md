# Stage 1: Data Acquisition — runs locally, not Colab

Windowed DEM(Digital Elevation Model) reads + crater catalog slicing don't need a GPU. Run this on your Mac; only the
`dem_crops/` output (small) gets uploaded to Colab later for training.

## Setup
```
pip install rasterio pandas
# if rasterio fails to build: brew install gdal, then retry
```

## Manual downloads

| File to place in `data/` | Get it from |
|---|---|
| `Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif` | [USGS product page](https://astrogeology.usgs.gov/search/map/moon_lro_lola_dem_118m) |
| `Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif` | [USGS product page](https://astrogeology.usgs.gov/search/map/mars_mgs_mola_mex_hrsc_blended_dem_global_200m) |
| `lunar_craters.csv` | [Moon Crater Database v1 Robbins](https://astrogeology.usgs.gov/search/map/moon_crater_database_v1_robbins) → "Download" (91.77 MB zip) → unzip → rename the CSV inside |
| `mars_craters.csv` | [Robbins Mars Crater DB, direct link](https://craters.sjrdesign.net/Catalog_Mars_Release_2020_1kmPlus_FullMorphData.csv.zip) → unzip → rename the file inside (it's tab-separated despite the `.csv` name — script auto-detects the delimiter) |

Note: the Astropedia bulk-download page for the Mars catalog (`.../Mars/Research/Craters/RobbinsCraterDatabase_20120821`)
404s — the real host for the Mars catalog is Robbins' own site above, not Astropedia.

## Folder layout
```
01_data_acquisition/
├── crop_regions.py
└── data/
    ├── Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif
    ├── Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif
    ├── lunar_craters.csv
    └── mars_craters.csv
```

## Run
```
cd 01_data_acquisition
python3 crop_regions.py
```
It pauses twice — once after loading each crater catalog — so you can confirm the printed column
names match `MOON_LAT_COL`/`MOON_LON_COL` and `MARS_LAT_COL`/`MARS_LON_COL` in the script before slicing.
Defaults are best guesses from the Robbins papers' column-naming conventions; not guaranteed to match
the exact release you downloaded.

## Output
```
dem_crops/
├── moon/  <region>.tif + <region>_craters.csv  (x3)
└── mars/  <region>.tif + <region>_craters.csv  (x3)
```

## Gotchas
- LOLA is int16 with `scale=0.5` — script applies it automatically via `ds.scales`.
- Confirm crater catalog column names against your actual file — Robbins naming shifts across releases.
- Mars catalog file is tab-separated despite the `.csv` extension; `pd.read_csv(..., sep=None, engine="python")` sniffs it.
- DEM longitude convention (0–360 vs −180/180) is auto-detected per file from `ds.bounds`.
- Regions chosen for terrain diversity, not geographic correspondence between Moon and Mars.