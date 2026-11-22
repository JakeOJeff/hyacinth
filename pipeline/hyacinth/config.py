"""Static configuration for the Hyacinth pipeline.

Everything that is a judgement call lives here so it can be tuned in one place.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
WEB_DATA_DIR = ROOT / "web" / "public" / "data"
RENDER_DIR = ROOT / "render"

# Southern Vembanad + Kuttanad, including the Thanneermukkom barrage and a strip of
# the northern lake above it. (lon_min, lat_min, lon_max, lat_max), WGS84.
# The Arabian Sea coast (~76.31 E) is excluded on purpose: the sea is not part
# of the OSM water polygons and is not part of the model.
BBOX = (76.28, 9.40, 76.55, 9.78)

# Simulation grid: UTM zone 43N, square cells in metres.
GRID_CRS = "EPSG:32643"
DX_M = 40.0

# Point used for the (spatially uniform) wind forcing: centre of the lake.
WIND_POINT = (9.59, 76.40)  # lat, lon

# Thanneermukkom barrage (bund). If OSM has a feature whose name matches
# BARRAGE_OSM_NAME it is used; otherwise this fallback line is drawn across the
# lake. Only cells that are water are converted, so overshooting onto land is
# harmless; the pipeline checks that the line actually seals the lake.
BARRAGE_OSM_NAME = "Thanneermukkom"
BARRAGE_FALLBACK_LINE = [(76.370, 9.6765), (76.430, 9.6765)]  # (lon, lat)
BARRAGE_WIDTH_M = 80.0

# The shutters are normally closed in the dry season (roughly mid-December to
# the end of March) to keep salt water out of the Kuttanad paddy. Override with
# HYACINTH_BARRAGE=open|closed when the real state is known.
def barrage_open(on: dt.date) -> bool:
    env = os.environ.get("HYACINTH_BARRAGE", "").strip().lower()
    if env in {"open", "1", "true"}:
        return True
    if env in {"closed", "0", "false"}:
        return False
    closed = (on.month == 12 and on.day >= 15) or on.month in (1, 2, 3)
    return not closed


# Line waterways (OSM ways without polygons) are buffered to these widths.
WATERWAY_WIDTH_M = {"river": 80.0, "canal": 50.0}
MIN_CHANNEL_CELLS = 2  # never narrower than this many cells

# Mats are detected at Sentinel-2's native 10 m on a grid FINE_FACTOR times finer
# than the simulation grid, inside a water mask eroded by one pixel so that
# bank vegetation mixed into a pixel is not mistaken for floating mats.
FINE_FACTOR = 4
assert abs(DX_M / FINE_FACTOR - 10.0) < 1e-6, "fine grid must be 10 m"

# Sentinel-2 mat detection.
S2_COLLECTION = "sentinel-2-l2a"
S2_MAX_SCENE_CLOUD = 60.0     # tile-level eo:cloud_cover prefilter, percent
S2_MIN_CLEAR_WATER = 0.70     # fraction of water cells that must be cloud-free
S2_LOOKBACK_DAYS = 30
NDVI_MAT = 0.45               # 10 m pixel inside water with NDVI above this = mat
MIN_MAT_PIXELS = 2            # per 40 m cell, to suppress single-pixel noise

# Jam aggregation for captions.
JAM_MIN_CELLS = 4             # ignore jam clusters smaller than this
TOP_SEGMENTS = 5

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
