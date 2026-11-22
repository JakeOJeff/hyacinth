"""A synthetic, Vembanad-shaped test domain that needs no network.

It is NOT real geography: names are placeholders and the output is flagged
``synthetic: true`` everywhere so it can never be mistaken for a forecast.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
from scipy import ndimage

from .io import BARRAGE, INFLOW, LAND, OUTLET, WATER

W, H, DX = 448, 640, 60.0


def _disk_line(mask, x0, y0, x1, y1, r):
    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    vx, vy = x1 - x0, y1 - y0
    L2 = vx * vx + vy * vy
    t = np.clip(((xx - x0) * vx + (yy - y0) * vy) / L2, 0, 1)
    d2 = (xx - (x0 + t * vx)) ** 2 + (yy - (y0 + t * vy)) ** 2
    mask |= d2 <= r * r


def build(start: dt.datetime, forecast_hours: int = 72, seed: int = 7):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)

    # Lake: a wobbly north-south body that narrows at the barrage.
    centre = W * 0.42 + 30 * np.sin(yy / H * 3.1) + 14 * np.sin(yy / 37.0)
    half = 70 + 40 * np.sin(yy / H * math.pi) ** 2
    half = np.where(np.abs(yy - H * 0.27) < 22, 26, half)  # the narrows
    half = np.where(yy > H * 0.86, half * np.clip((H - yy) / (H * 0.14), 0.0, 1.0), half)
    lake = np.abs(xx - centre) < half

    # An island (Pathiramanal-like) and some ragged shoreline.
    lake &= ((xx - W * 0.45) ** 2 / 18 ** 2 + (yy - H * 0.42) ** 2 / 26 ** 2) > 1
    noise = ndimage.gaussian_filter(rng.standard_normal((H, W)), 6)
    lake &= noise < 0.09

    water = lake.copy()
    # Canals and rivers out to the east / south edges.
    _disk_line(water, W * 0.55, H * 0.70, W - 1, H * 0.62, 3)   # river to east edge
    _disk_line(water, W * 0.50, H * 0.82, W * 0.80, H - 1, 3)   # river to south edge
    _disk_line(water, W * 0.40, H * 0.60, W * 0.62, H * 0.93, 2)  # canal network
    _disk_line(water, W * 0.62, H * 0.93, W * 0.95, H * 0.97, 2)
    _disk_line(water, W * 0.30, H * 0.75, W * 0.12, H * 0.92, 2)  # canal to the west town
    _disk_line(water, W * 0.42, H * 0.05, W * 0.42, 0, 10)        # lake continues north

    lab, n = ndimage.label(water)
    sizes = ndimage.sum(water, lab, index=np.arange(1, n + 1))
    water = lab == 1 + int(np.argmax(sizes))

    cells = np.where(water, WATER, LAND).astype(np.uint8)
    bar = (np.abs(yy - H * 0.27) <= 1) & water
    cells[bar] = BARRAGE
    cells[0, :][cells[0, :] == WATER] = OUTLET
    cells[:, -1][cells[:, -1] == WATER] = INFLOW
    cells[-1, :][cells[-1, :] == WATER] = INFLOW

    # Hyacinth: a few smooth patches in the southern basin and canals.
    blobs = ndimage.gaussian_filter(rng.standard_normal((H, W)), 9)
    patchy = (blobs > 0.06) & (yy > H * 0.45)
    cover = np.clip((blobs - 0.06) / 0.08, 0, 1)
    open_water = cells == WATER
    seed_map = np.where(patchy & open_water, 64 + 191 * cover, 0).astype(np.uint8)

    # Wind: diurnal sea breeze from the WSW by day, weak land breeze at night.
    hours = forecast_hours + 1
    t = np.arange(hours)
    local = (start.hour + 5.5 + t) % 24
    day = np.clip(np.sin((local - 7) / 24 * 2 * math.pi), 0, None)
    speed = 1.5 + 4.5 * day + 0.5 * rng.standard_normal(hours).cumsum() / np.sqrt(hours)
    speed = np.clip(speed, 0.3, None)
    dir_from = np.where(day > 0.1, 250.0, 60.0) + 15 * np.sin(t / 7.0)
    r = np.deg2rad(dir_from)
    u, v = -speed * np.sin(r), -speed * np.cos(r)

    iso = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    wind = {
        "start": iso,
        "forecast_start": iso,
        "forecast_hours": forecast_hours,
        "step_hours": 1,
        "u": [round(float(x), 3) for x in u],
        "v": [round(float(x), 3) for x in v],
        "dx_m": DX,
        "barrage_open": False,
        "source": "synthetic",
        "scene_date": None,
        "synthetic": True,
    }

    # Placeholder names for the caption pipeline.
    places = [
        {"name": "North Jetty (synthetic)", "name_ml": None, "cx": W * 0.42, "cy": H * 0.12},
        {"name": "Island Point (synthetic)", "name_ml": None, "cx": W * 0.45, "cy": H * 0.42},
        {"name": "East Landing (synthetic)", "name_ml": None, "cx": W * 0.85, "cy": H * 0.63},
        {"name": "Canal Junction (synthetic)", "name_ml": None, "cx": W * 0.62, "cy": H * 0.92},
        {"name": "West Town (synthetic)", "name_ml": None, "cx": W * 0.14, "cy": H * 0.90},
        {"name": "South Basin (synthetic)", "name_ml": None, "cx": W * 0.45, "cy": H * 0.75},
    ]
    pc = np.array([(p["cx"], p["cy"]) for p in places])
    d = (xx[..., None] + 0.5 - pc[:, 0]) ** 2 + (yy[..., None] + 0.5 - pc[:, 1]) ** 2
    place_id = np.where(cells != LAND, d.argmin(-1), -1).astype(np.int16)
    labels = {
        "waterway_id": np.full((H, W), -1, np.int16),
        "place_id": place_id,
        "waterways": [],
        "places": [{k: p[k] for k in ("name", "name_ml")} for p in places],
    }
    grid = {"crs": "synthetic", "transform": [DX, 0, 0, 0, -DX, 0], "width": W, "height": H, "dx": DX}
    return cells, seed_map, wind, labels, grid
