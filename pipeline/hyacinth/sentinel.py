"""Sentinel-2 L2A mat detection via the Microsoft Planetary Computer STAC API.

Hyacinth mats are floating vegetation, so inside the (OSM) water mask they show
up as high NDVI. Mat-covered water fails an NDWI test, which is why the water
domain comes from OSM rather than from NDWI. Detection runs at the native 10 m
on a water mask eroded off the banks; each 40 m simulation cell gets the
fraction of its area covered by mats.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy import ndimage

from . import config
from .grid import Grid
from .io import INFLOW, WATER

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
CLEAR_SCL = (2, 4, 5, 6, 7)  # dark area, vegetation, bare, water, unclassified
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
}


@dataclass
class Scene:
    when: dt.datetime
    seed: np.ndarray        # uint8 [H, W]: fraction of the cell covered by mats, 0..255
    clear: np.ndarray       # bool [H, W]: cell's interior water was mostly cloud-free
    clear_water: float      # fraction of interior water pixels that were cloud-free
    mat_cells: int
    tiles: list[str]


def _catalog():
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def _read(href: str, grid: Grid, resampling: Resampling) -> np.ndarray:
    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src, WarpedVRT(
        src, crs=grid.crs, transform=grid.transform, width=grid.width, height=grid.height,
        resampling=resampling, src_nodata=0, nodata=0,
    ) as vrt:
        return vrt.read(1)


def _reflectance(dn: np.ndarray, baseline: str | None) -> np.ndarray:
    """L2A DN -> surface reflectance. Processing baseline 04.00+ (25 Jan 2022)
    adds a +1000 offset to every band (BOA_ADD_OFFSET = -1000)."""
    offset = 0.0
    try:
        if baseline is not None and float(baseline) >= 4.0:
            offset = -1000.0
    except ValueError:
        pass
    r = (dn.astype(np.float32) + offset) / 10000.0
    r[dn == 0] = np.nan
    return r


def _group_by_date(items) -> dict[dt.date, list]:
    groups: dict[dt.date, list] = defaultdict(list)
    for it in items:
        groups[it.datetime.date()].append(it)
    return groups


def _process(items, grid: Grid, cells: np.ndarray, water_fine: np.ndarray) -> Scene:
    """Detect mats at 10 m inside the bank-eroded water mask, then aggregate
    to the fraction of each simulation cell that is covered."""
    k = config.FINE_FACTOR
    fine = grid.subdivide(k)
    red = np.full(fine.shape, np.nan, np.float32)
    nir = np.full(fine.shape, np.nan, np.float32)
    scl = np.zeros(fine.shape, np.uint8)
    tiles = []
    # Clearest tile first; later tiles only fill gaps.
    for it in sorted(items, key=lambda i: i.properties.get("eo:cloud_cover", 100)):
        gap = np.isnan(red)
        if not gap.any():
            break
        tiles.append(it.properties.get("s2:mgrs_tile", it.id))
        base = it.properties.get("s2:processing_baseline")
        r = _reflectance(_read(it.assets["B04"].href, fine, Resampling.nearest), base)
        n = _reflectance(_read(it.assets["B08"].href, fine, Resampling.nearest), base)
        s = _read(it.assets["SCL"].href, fine, Resampling.nearest)
        fill = gap & ~np.isnan(r) & ~np.isnan(n)
        red[fill] = r[fill]
        nir[fill] = n[fill]
        scl[fill] = s[fill]

    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red)
    clear = np.isin(scl, CLEAR_SCL) & np.isfinite(ndvi)
    # One 10 m pixel off every bank: those pixels mix in palms and paddy.
    interior = ndimage.binary_erosion(water_fine, structure=ndimage.generate_binary_structure(2, 1))
    mat = interior & clear & (ndvi > config.NDVI_MAT)

    def per_cell(a: np.ndarray) -> np.ndarray:
        return a.reshape(grid.height, k, grid.width, k).sum(axis=(1, 3))

    n_mat = per_cell(mat)
    n_int = per_cell(interior)
    n_clear = per_cell(interior & clear)

    sim_water = (cells == WATER) | (cells == INFLOW)
    cover = n_mat / float(k * k)
    seed = np.where(sim_water & (n_mat >= config.MIN_MAT_PIXELS), np.round(255 * cover), 0).astype(np.uint8)
    cell_clear = (n_int > 0) & (n_clear * 2 >= n_int)
    clear_frac = float(n_clear.sum() / max(1, n_int.sum()))
    return Scene(items[0].datetime, seed, cell_clear, clear_frac, int((seed > 0).sum()), tiles)


def latest_scene(grid: Grid, cells: np.ndarray, water_fine: np.ndarray, until: dt.date,
                 lookback_days=config.S2_LOOKBACK_DAYS) -> Scene:
    """Newest scene in the window whose water is mostly cloud-free."""
    start = until - dt.timedelta(days=lookback_days)
    search = _catalog().search(
        collections=[config.S2_COLLECTION],
        bbox=config.BBOX,
        datetime=f"{start.isoformat()}/{until.isoformat()}T23:59:59Z",
        query={"eo:cloud_cover": {"lt": config.S2_MAX_SCENE_CLOUD}},
    )
    groups = _group_by_date(search.item_collection())
    if not groups:
        raise RuntimeError(f"no Sentinel-2 scenes under {config.S2_MAX_SCENE_CLOUD}% cloud "
                           f"between {start} and {until}")
    for day in sorted(groups, reverse=True):
        print(f"[s2] trying {day} ({len(groups[day])} tile(s))")
        scene = _process(groups[day], grid, cells, water_fine)
        print(f"[s2]   clear water {scene.clear_water:.0%}, mat cells {scene.mat_cells}")
        if scene.clear_water >= config.S2_MIN_CLEAR_WATER:
            return scene
    raise RuntimeError(f"no scene with >= {config.S2_MIN_CLEAR_WATER:.0%} clear water since {start} "
                       "(monsoon cloud?). Sentinel-1 radar is the fallback; not implemented yet.")


def scene_on(grid: Grid, cells: np.ndarray, water_fine: np.ndarray, day: dt.date) -> Scene:
    search = _catalog().search(
        collections=[config.S2_COLLECTION],
        bbox=config.BBOX,
        datetime=f"{day.isoformat()}T00:00:00Z/{day.isoformat()}T23:59:59Z",
    )
    items = list(search.item_collection())
    if not items:
        raise RuntimeError(f"no Sentinel-2 scene on {day}")
    return _process(items, grid, cells, water_fine)


def clear_dates(start: dt.date, end: dt.date, max_cloud: float = 20.0) -> list[tuple[dt.date, float]]:
    """Dates with low tile-level cloud cover, for picking hindcast pairs."""
    search = _catalog().search(
        collections=[config.S2_COLLECTION],
        bbox=config.BBOX,
        datetime=f"{start.isoformat()}/{end.isoformat()}T23:59:59Z",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    groups = _group_by_date(search.item_collection())
    return sorted((d, float(np.mean([i.properties["eo:cloud_cover"] for i in its]))) for d, its in groups.items())
