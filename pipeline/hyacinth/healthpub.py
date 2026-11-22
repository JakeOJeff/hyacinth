"""Publish the lake-health dataset for the site and the assistant.

web/public/data/health/
  health.json   zones, every usable Sentinel-2 pass with per-zone metrics and a
                relative health index, debris sightings, daily geostationary
                conditions + weather context, and the latest "now" reading
  hotspots.png  40 m map: how often each cell had mats (green) / possible debris
                (magenta) across all clear passes
  latest.png    20 m map of the most recent clear pass (mats, possible debris, cloud)

The health index is RELATIVE: for each zone, each indicator is ranked against
that zone's own record since HEALTH_START (0 = best seen, 1 = worst seen); the
index is 100 x (1 - mean rank). It says "better or worse than usual here", not
whether the water meets any standard.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from PIL import Image
from scipy import ndimage

from . import config, geo
from .health import HEALTH_DIR, HEALTH_START, K, METRICS, Domain
from .io import read_json, write_json

OUT_DIR = config.WEB_DATA_DIR / "health"
MIN_HISTORY = 6

METRIC_INFO = {
    "health": {"label": "Health index", "unit": "/100", "higher_is": "better",
               "desc": "Relative condition vs this zone's own record since Jan 2025 (100 = best seen)."},
    "hyacinth_pct": {"label": "Hyacinth cover", "unit": "% of open water", "higher_is": "worse",
                     "desc": "Share of cloud-free water covered by floating mats (Sentinel-2 NDVI)."},
    "debris_ha_per_km2": {"label": "Possible debris", "unit": "ha per km²", "higher_is": "worse",
                          "desc": "Small floating non-vegetation patches (Floating Debris Index). Low confidence per pass; recurring spots matter."},
    "chlorophyll": {"label": "Algae (chlorophyll proxy)", "unit": "NDCI", "higher_is": "worse",
                    "desc": "Normalised Difference Chlorophyll Index over open water; higher = more algae."},
    "turbidity": {"label": "Turbidity", "unit": "NDTI", "higher_is": "worse",
                  "desc": "Normalised Difference Turbidity Index over open water; higher = muddier."},
}


def _ranks(values: list[float | None]) -> list[float | None]:
    """Empirical percentile of each value within its own series (0 best .. 1 worst)."""
    vals = np.array([v for v in values if v is not None], float)
    if len(vals) < MIN_HISTORY:
        return [None] * len(values)
    srt = np.sort(vals)
    out = []
    for v in values:
        if v is None:
            out.append(None)
        else:
            lo = np.searchsorted(srt, v, "left")
            hi = np.searchsorted(srt, v, "right")
            out.append(((lo + hi) / 2) / len(srt))
    return out


def _geo_cached(start: dt.date, end: dt.date) -> tuple[dict, dict]:
    """Daily satellite + weather, cached; only the last ten days and missing days are refetched."""
    p = HEALTH_DIR / "geo.json"
    cache = read_json(p) if p.exists() else {"sat": {}, "wx": {}}
    have = sorted(cache["sat"])
    fetch_from = start if not have else min(dt.date.fromisoformat(have[-1]) - dt.timedelta(days=10), end)
    try:
        cache["sat"].update(geo.satellite_daily(fetch_from, end))
    except Exception as e:
        print(f"[health] geostationary fetch failed: {e}")
    have_wx = sorted(cache["wx"])
    wx_from = start if not have_wx else min(dt.date.fromisoformat(have_wx[-1]) - dt.timedelta(days=10), end)
    cache["wx"].update(geo.weather_daily(wx_from, end))
    write_json(cache, p)
    return cache["sat"], cache["wx"]


def _palette_image(rgb: np.ndarray, path) -> None:
    Image.fromarray(rgb.astype(np.uint8), "RGB").save(path, optimize=True)


LAND = np.array([31, 28, 25])
WATER = np.array([40, 56, 66])
MAT = np.array([150, 196, 92])
DEBRIS = np.array([236, 72, 170])
CLOUD = np.array([120, 120, 128])


def _hotspot_map(dom: Domain) -> None:
    z = np.load(HEALTH_DIR / "accum.npz")
    obs, hy, deb = z["obs"].astype(float), z["hy"].astype(float), z["deb"].astype(float)
    water40 = dom.water20.reshape(dom.grid.height, K, dom.grid.width, K).any(axis=(1, 3))
    rgb = np.where(water40[..., None], WATER, LAND).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        f_hy = np.where(obs >= 5, hy / obs, 0.0)
        f_deb = np.where(obs >= 5, deb / obs, 0.0)
    a = np.clip(f_hy / 0.6, 0, 1)[..., None]  # 60 %+ of passes = full colour
    rgb = rgb * (1 - a) + MAT * a
    d = ndimage.binary_dilation(f_deb >= 0.08, iterations=1)  # recurring possible debris
    rgb[d] = DEBRIS
    _palette_image(rgb, OUT_DIR / "hotspots.png")


def _latest_map(dom: Domain) -> str | None:
    p = HEALTH_DIR / "latest.npz"
    if not p.exists():
        return None
    z = np.load(p)
    water20 = dom.water20
    rgb = np.where(water20[..., None], WATER, LAND).astype(float)
    cloud = dom.interior & ~z["clear"]
    rgb[cloud] = rgb[cloud] * 0.4 + CLOUD * 0.6
    rgb[z["hy"]] = MAT
    rgb[ndimage.binary_dilation(z["deb"], iterations=2)] = DEBRIS
    _palette_image(rgb, OUT_DIR / "latest.png")
    return str(z["date"])


def publish() -> dict:
    state_p = HEALTH_DIR / "state.json"
    if not state_p.exists():
        raise RuntimeError("no health dataset yet: run `python -m hyacinth health` first")
    state = read_json(state_p)
    dom = Domain()
    zones = state["zones"]
    passes = state["passes"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    nz, npass = len(zones), len(passes)
    series = {m: [[(p["zones"][z] or {}).get(m) for p in passes] for z in range(nz)] for m in METRICS}
    clear = [[(p["zones"][z] or {}).get("clear") for p in passes] for z in range(nz)]
    ranks = {m: [_ranks(series[m][z]) for z in range(nz)] for m in METRICS}
    health = []
    for z in range(nz):
        row = []
        for i in range(npass):
            rs = [ranks[m][z][i] for m in METRICS if ranks[m][z][i] is not None]
            row.append(round(100 * (1 - sum(rs) / len(rs))) if len(rs) >= 2 else None)
        health.append(row)
    series["health"] = health

    # Hotspots: per-zone averages over all passes.
    hotspots = []
    for z in range(nz):
        obs = sum(1 for c in clear[z] if c is not None)
        deb_hits = sum(1 for p in passes for s in p["sightings"] if s["zone"] == z and z != 0)
        def mean(m):
            v = [x for x in series[m][z] if x is not None]
            return round(float(np.mean(v)), 4) if v else None
        hotspots.append({
            "zone": z, "passes_observed": obs,
            "hyacinth_pct_mean": mean("hyacinth_pct"), "debris_ha_per_km2_mean": mean("debris_ha_per_km2"),
            "chlorophyll_mean": mean("chlorophyll"), "turbidity_mean": mean("turbidity"),
            "debris_sightings": deb_hits if z else sum(len(p["sightings"]) for p in passes),
        })

    sightings = [
        {"date": p["date"], **s} for p in passes for s in p["sightings"]
    ]

    start = HEALTH_START
    end = dt.date.today()
    sat, wx = _geo_cached(start, end)
    days = []
    d = start
    while d <= end:
        days.append(d.isoformat())
        d += dt.timedelta(days=1)
    daily = {
        "date": days,
        "cloudiness": [sat.get(x, {}).get("cloudiness") for x in days],
        "insolation_kwh": [sat.get(x, {}).get("insolation_kwh") for x in days],
        "rain_mm": [wx.get(x, {}).get("rain_mm") for x in days],
        "tmax_c": [wx.get(x, {}).get("tmax_c") for x in days],
    }
    try:
        now = geo.satellite_now()
    except Exception as e:
        print(f"[health] geostationary now failed: {e}")
        now = None

    _hotspot_map(dom)
    latest_date = _latest_map(dom)

    out = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "period": [start.isoformat(), end.isoformat()],
        "grid": {"width": dom.grid.width, "height": dom.grid.height, "dx_m": dom.grid.dx},
        "sources": {
            "passes": "Copernicus Sentinel-2 L2A (ESA) via Microsoft Planetary Computer, analysed at 20 m",
            "daily_cloudiness": "Geostationary satellites (EUMETSAT Meteosat IODC / JMA Himawari) via Open-Meteo satellite radiation API",
            "daily_weather": "Open-Meteo: ERA5 reanalysis, last days from forecast models (context, not satellite)",
            "zones": "OpenStreetMap places (© OpenStreetMap contributors)",
        },
        "caveats": [
            "Indicators come from surface reflectance and are relative; they are not laboratory water-quality measurements.",
            "Possible debris is low confidence per pass: at 20 m, small young hyacinth clumps can look similar. Spots that recur across passes are more reliable.",
            "Optical satellites cannot see through cloud; the monsoon leaves long gaps.",
            "Only the lake and channels wider than about 100 m are analysed; narrow canals mix with bank vegetation at 20 m and are excluded.",
            "The health index ranks each zone against its own record since Jan 2025; it is not a regulatory standard.",
        ],
        "metrics": METRIC_INFO,
        "zones": [{k: z[k] for k in ("id", "name", "name_ml", "water_ha", "x", "y")} for z in zones],
        "passes": {
            "date": [p["date"] for p in passes],
            "time": [p["time"] for p in passes],
            "clear": [p["clear"] for p in passes],
        },
        "series": series,
        "hotspots": hotspots,
        "sightings": sightings,
        "daily": daily,
        "now": now,
        "latest_pass": latest_date,
        "images": {"hotspots": "hotspots.png", "latest": "latest.png" if latest_date else None},
    }
    write_json(out, OUT_DIR / "health.json")
    print(f"[health] published {npass} passes x {nz} zones, {len(sightings)} debris sightings -> {OUT_DIR}")
    return out
