"""Lake-health dataset: every usable Sentinel-2 pass since HEALTH_START, per zone.

For each date the lake was mostly cloud-free, at 20 m inside the bank-eroded
OSM water mask:

  hyacinth   NDVI > NDVI_MAT                       floating vegetation mats
  debris     POSSIBLE floating debris: Floating Debris Index (Biermann et al.
             2020) well above this pass's own open water (FDI_MARGIN over its
             99.5th percentile; turbidity and atmosphere shift the baseline),
             NDVI near zero, >= 40 m from banks and from mats, in patches of
             0.08-1 ha. At 20 m, small young hyacinth clumps can look the
             same, so single sightings are low confidence; patches that recur
             across passes are the useful signal.
  ndci       (B05 - B04) / (B05 + B04) over open water: chlorophyll-a proxy
             (Mishra & Mishra 2012), higher = more algae
  ndti       (B04 - B03) / (B04 + B03) over open water: turbidity proxy,
             higher = muddier water

All are relative indicators from surface reflectance, not lab measurements.
Only water at least 40 m from a bank is analysed (the lake and wide channels):
narrow canals mix with bank vegetation at 20 m and are left out.
Results are stored incrementally in data/health/ so the daily run only adds
the newest pass. Zones are the OSM places the forecast already uses.
"""

from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import planetary_computer
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from scipy import ndimage

from . import config
from .dataset import load_labels
from .grid import Grid
from .io import INFLOW, WATER, read_json, read_mask, write_json
from .sentinel import CLEAR_SCL, _catalog, _group_by_date, _read, _reflectance

HEALTH_DIR = config.DATA_DIR / "health"
HEALTH_START = dt.date(2025, 1, 1)
K = 2                      # 20 m analysis pixels per 40 m cell edge
PX_HA = (config.DX_M / K) ** 2 / 1e4
MIN_CLEAR = 0.30           # lake-wide clear-water fraction to keep a pass
ZONE_MIN_CLEAR = 0.40      # per-zone clear fraction to report its metrics
ZONE_MIN_CELLS = 40        # 40 m interior-water cells for a place to be a zone
FDI_MARGIN = 0.02          # above the pass's open-water FDI 99.5th percentile
DEBRIS_NDVI = (-0.05, 0.20)
MIN_DEBRIS_PX = 2          # 0.08 ha; single pixels are mostly noise
MAX_DEBRIS_PX = 25         # 1 ha; bigger "patches" are vegetation or shallows
MAX_SIGHTINGS_PER_PASS = 40
BANDS = ("B03", "B04", "B05", "B06", "B08", "B11")
# Band centre wavelengths (nm) for the FDI baseline.
L_RED, L_NIR, L_SWIR1 = 664.6, 832.8, 1613.7

METRICS = ("hyacinth_pct", "debris_ha_per_km2", "chlorophyll", "turbidity")


def _base_dir() -> Path:
    d = config.DATA_DIR / "latest"
    if not (d / "mask.png").exists() or not (d / "labels.npz").exists():
        raise RuntimeError("run `python -m hyacinth fetch` once first (needs data/latest mask and labels)")
    return d


class Domain:
    """Water mask and zones on the 20 m analysis grid."""

    def __init__(self):
        base = _base_dir()
        self.grid = Grid.load(base / "grid.json")
        self.fine = self.grid.subdivide(K)
        cells = read_mask(base / "mask.png")
        water40 = (cells == WATER) | (cells == INFLOW)
        water20 = np.kron(water40, np.ones((K, K), bool))
        self.water20 = water20
        # Only water >= 40 m from any bank: OSM canals are widened to 80 m for the
        # simulation, so at 20 m their "water" pixels are mostly bank palms and
        # paddy. Narrow canals are therefore excluded from the health record.
        self.interior = ndimage.binary_erosion(water20, structure=ndimage.generate_binary_structure(2, 1), iterations=2)
        self.deep = ndimage.binary_erosion(self.interior, iterations=1)  # >= 60 m off banks
        labels = load_labels(base)
        place40 = labels["place_id"].astype(np.int32)
        place20 = np.kron(place40, np.ones((K, K), np.int32))
        # Zones: places with enough interior water. Zone 0 is the whole lake.
        counts = np.bincount(place20[self.interior & (place20 >= 0)], minlength=len(labels["places"])) / (K * K)
        keep = [int(p) for p in np.nonzero(counts >= ZONE_MIN_CELLS)[0]]
        self.zones = [{"id": 0, "place": -1, "name": "Whole lake", "name_ml": "മുഴുവൻ കായൽ"}]
        for i, p in enumerate(keep, start=1):
            pl = labels["places"][p]
            self.zones.append({"id": i, "place": p, "name": pl["name"], "name_ml": pl.get("name_ml")})
        zmap = np.full(len(labels["places"]), -1, np.int32)
        for z in self.zones[1:]:
            zmap[z["place"]] = z["id"]
        self.zone20 = np.where(place20 >= 0, zmap[np.clip(place20, 0, None)], -1)
        self.zone20[~self.interior] = -1
        for z in self.zones:
            m = self.interior if z["id"] == 0 else self.zone20 == z["id"]
            z["water_ha"] = round(float(m.sum() * PX_HA), 1)
            ys, xs = np.nonzero(m)
            # Centroid in 40 m sim-grid cell coordinates (what the web map uses).
            z["x"] = round(float(xs.mean() / K), 1) if len(xs) else None
            z["y"] = round(float(ys.mean() / K), 1) if len(ys) else None

    def lonlat(self, rows: np.ndarray, cols: np.ndarray) -> tuple[list[float], list[float]]:
        xs, ys = self.fine.transform * (cols + 0.5, rows + 0.5)
        return warp_transform(self.fine.crs, "EPSG:4326", list(np.atleast_1d(xs)), list(np.atleast_1d(ys)))


def _mosaic(items, fine: Grid) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Reflectance per band + SCL on the 20 m grid; clearest tile first, others fill gaps."""
    out = {b: np.full(fine.shape, np.nan, np.float32) for b in BANDS}
    scl = np.zeros(fine.shape, np.uint8)
    tiles = []
    for it in sorted(items, key=lambda i: i.properties.get("eo:cloud_cover", 100)):
        gap = np.isnan(out["B04"])
        if not gap.any():
            break
        tiles.append(it.properties.get("s2:mgrs_tile", it.id))
        base = it.properties.get("s2:processing_baseline")
        # Network-bound COG reads: fetch the bands of a tile in parallel.
        with ThreadPoolExecutor(max_workers=len(BANDS) + 1) as pool:
            futs = {b: pool.submit(_read, it.assets[b].href, fine, Resampling.average) for b in BANDS}
            f_scl = pool.submit(_read, it.assets["SCL"].href, fine, Resampling.nearest)
            refl = {b: _reflectance(futs[b].result(), base) for b in BANDS}
            s = f_scl.result()
        fill = gap & np.all([~np.isnan(refl[b]) for b in BANDS], axis=0)
        for b in BANDS:
            out[b][fill] = refl[b][fill]
        scl[fill] = s[fill]
    return out, scl, tiles


def _ratio(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a - b) / (a + b)


def analyse(items, dom: Domain) -> dict | None:
    r, scl, tiles = _mosaic(items, dom.fine)
    ndvi = _ratio(r["B08"], r["B04"])
    ndci = _ratio(r["B05"], r["B04"])
    ndti = _ratio(r["B04"], r["B03"])
    with np.errstate(invalid="ignore"):
        baseline = r["B06"] + (r["B11"] - r["B06"]) * ((L_NIR - L_RED) / (L_SWIR1 - L_RED)) * 10.0
        fdi = r["B08"] - baseline
        clear = dom.interior & np.isin(scl, CLEAR_SCL) & np.isfinite(ndvi) & np.isfinite(fdi)
        hy = clear & (ndvi > config.NDVI_MAT)
        near_hy = ndimage.binary_dilation(hy, iterations=2)
        open_w = clear & ~near_hy & (ndvi < 0.0)
        fdi_ref = float(np.nanpercentile(fdi[open_w], 99.5)) if open_w.sum() > 1000 else np.inf
        deb = (clear & dom.deep & ~near_hy & (fdi > fdi_ref + FDI_MARGIN)
               & (ndvi > DEBRIS_NDVI[0]) & (ndvi <= DEBRIS_NDVI[1]))
        open_w &= ~ndimage.binary_dilation(deb, iterations=1)

    lake_clear = float(clear.sum() / max(1, dom.interior.sum()))
    if lake_clear < MIN_CLEAR:
        return {"clear": round(lake_clear, 3), "tiles": tiles, "accepted": False}

    # Debris patches at 20 m: 8-connected components of at least MIN_DEBRIS_PX.
    lab, n = ndimage.label(deb, structure=np.ones((3, 3), bool))
    deb_kept = np.zeros_like(deb)
    sightings = []
    if n:
        sizes = ndimage.sum(deb, lab, index=np.arange(1, n + 1))
        coms = ndimage.center_of_mass(deb, lab, index=np.arange(1, n + 1))
        order = np.argsort(-sizes)
        for i in order:
            if sizes[i] < MIN_DEBRIS_PX:
                break
            if sizes[i] > MAX_DEBRIS_PX:
                continue
            deb_kept |= lab == (i + 1)
            if len(sightings) < MAX_SIGHTINGS_PER_PASS:
                ry, rx = coms[i]
                zid = int(dom.zone20[int(round(ry)), int(round(rx))])
                lon, lat = dom.lonlat(np.array([ry]), np.array([rx]))
                sightings.append({
                    "zone": max(zid, 0), "ha": round(float(sizes[i] * PX_HA), 2),
                    "lon": round(lon[0], 5), "lat": round(lat[0], 5),
                    "x": round(rx / K, 1), "y": round(ry / K, 1),
                })

    zones = []
    for z in dom.zones:
        m = dom.interior if z["id"] == 0 else dom.zone20 == z["id"]
        n_int = int(m.sum())
        n_clear = int((m & clear).sum())
        frac = n_clear / max(1, n_int)
        if frac < ZONE_MIN_CLEAR:
            zones.append(None)
            continue
        ow = m & open_w
        clear_km2 = n_clear * PX_HA / 100.0
        zones.append({
            "clear": round(frac, 3),
            "hyacinth_pct": round(100.0 * (m & hy).sum() / n_clear, 2),
            "debris_ha_per_km2": round(float((m & deb_kept).sum() * PX_HA / clear_km2), 3),
            "chlorophyll": round(float(np.nanmedian(ndci[ow])), 4) if ow.sum() >= 20 else None,
            "turbidity": round(float(np.nanmedian(ndti[ow])), 4) if ow.sum() >= 20 else None,
        })

    # 40 m accumulators for frequency maps.
    def per_cell(a):
        return a.reshape(dom.grid.height, K, dom.grid.width, K).sum(axis=(1, 3))

    obs40 = per_cell(clear) * 2 >= K * K
    return {
        "clear": round(lake_clear, 3), "tiles": tiles, "accepted": True, "zones": zones, "sightings": sightings,
        "acc": {
            "obs": obs40,
            "hy": obs40 & (per_cell(hy) > 0),
            "deb": obs40 & (per_cell(deb_kept) > 0),
        },
        "masks": {"hy": hy, "deb": deb_kept, "clear": clear},
    }


# ---------------------------------------------------------------- storage

def _load_state() -> dict:
    p = HEALTH_DIR / "state.json"
    return read_json(p) if p.exists() else {"version": 1, "seen": {}, "passes": []}


def _load_acc(shape) -> dict[str, np.ndarray]:
    p = HEALTH_DIR / "accum.npz"
    if p.exists():
        z = np.load(p)
        if z["obs"].shape == shape:
            return {k: z[k].astype(np.int32) for k in ("obs", "hy", "deb")}
    return {k: np.zeros(shape, np.int32) for k in ("obs", "hy", "deb")}


def build(start: dt.date = HEALTH_START, end: dt.date | None = None, limit: int | None = None) -> int:
    """Process every not-yet-seen Sentinel-2 date in [start, end]. Returns passes added."""
    end = end or dt.date.today()
    HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    dom = Domain()
    state = _load_state()
    if state.get("zones") and [z["name"] for z in state["zones"]] != [z["name"] for z in dom.zones]:
        raise RuntimeError("zones changed since the dataset was built; delete data/health/ and rebuild")
    state["zones"] = dom.zones
    acc = _load_acc(dom.grid.shape)

    search = _catalog().search(
        collections=[config.S2_COLLECTION], bbox=config.BBOX,
        datetime=f"{start.isoformat()}/{end.isoformat()}T23:59:59Z",
        query={"eo:cloud_cover": {"lt": 85}},
    )
    groups = _group_by_date(search.item_collection())
    todo = [d for d in sorted(groups) if d.isoformat() not in state["seen"]]
    if limit:
        todo = todo[:limit]
    print(f"[health] {len(groups)} candidate dates, {len(todo)} new")
    added = 0
    for i, day in enumerate(todo, start=1):
        try:
            # Search-time SAS signatures expire after about an hour; re-sign before reading.
            res = analyse([planetary_computer.sign(it) for it in groups[day]], dom)
        except Exception as e:  # one bad tile must not stop a two-year backfill
            print(f"[health] {day}: failed ({type(e).__name__}: {e}); will retry next run")
            continue
        when = min(it.datetime for it in groups[day])
        state["seen"][day.isoformat()] = "ok" if res["accepted"] else "cloudy"
        print(f"[health] {i}/{len(todo)} {day} clear {res['clear']:.0%}" + ("" if res["accepted"] else " (skipped)"))
        if res["accepted"]:
            state["passes"].append({
                "date": day.isoformat(), "time": when.isoformat().replace("+00:00", "Z"),
                "clear": res["clear"], "tiles": res["tiles"], "zones": res["zones"], "sightings": res["sightings"],
            })
            for k in acc:
                acc[k] += res["acc"][k]
            if not any(p["date"] > day.isoformat() for p in state["passes"]):
                np.savez_compressed(HEALTH_DIR / "latest.npz", date=day.isoformat(), **res["masks"])
            added += 1
        state["passes"].sort(key=lambda p: p["date"])
        # Checkpoint every date so an interrupted backfill resumes where it stopped.
        write_json(state, HEALTH_DIR / "state.json")
        np.savez_compressed(HEALTH_DIR / "accum.npz", **acc)
    return added
