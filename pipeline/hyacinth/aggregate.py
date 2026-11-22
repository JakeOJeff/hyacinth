"""Turn the renderer's per-cell jam output into named, ranked facts.

Everything a caption may say comes from here, in code; the language model only
phrases these facts.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import config
from .dataset import load_labels, read_bin
from .io import read_json, read_mask
from .wind import summary as wind_summary

WEEKDAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAYS_ML = ["തിങ്കളാഴ്ച", "ചൊവ്വാഴ്ച", "ബുധനാഴ്ച", "വ്യാഴാഴ്ച", "വെള്ളിയാഴ്ച", "ശനിയാഴ്ച", "ഞായറാഴ്ച"]

# Cells jammed within the first hour were already covered when the run began.
EXISTING_HOURS = 1.0


def _parse(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def jam_facts(d: Path) -> dict:
    out = d / "out"
    stats = read_json(out / "stats.json")
    wind = read_json(d / "wind.json")
    meta = read_json(d / "meta.json") if (d / "meta.json").exists() else {}
    shape = (stats["height"], stats["width"])
    eta = read_bin(out / "eta.bin", shape)
    cells = read_mask(d / "mask.png")
    if cells.shape != shape:
        raise ValueError("mask.png does not match the render output")
    labels = load_labels(d)
    places, waterways = labels["places"], labels["waterways"]

    ha = stats["dx_m"] ** 2 / 1e4
    start = _parse(stats["forecast_start"])

    jammed = eta >= 0
    lab, n = ndimage.label(jammed, structure=np.ones((3, 3), bool))
    groups: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == k
        if m.sum() < config.JAM_MIN_CELLS:
            continue
        pid = labels["place_id"][sl][m]
        wid = labels["waterway_id"][sl][m]
        place = int(np.bincount(pid[pid >= 0]).argmax()) if (pid >= 0).any() else -1
        ww = int(np.bincount(wid[wid >= 0]).argmax()) if (wid >= 0).any() else -1
        ys, xs = np.nonzero(m)
        groups[(place, ww)].append(np.stack([ys + sl[0].start, xs + sl[1].start]))

    segments = []
    for (place, ww), parts in groups.items():
        yx = np.concatenate(parts, axis=1)
        e = eta[yx[0], yx[1]]
        new = e >= EXISTING_HOURS
        new_ha = float(new.sum() * ha)
        existing_ha = float((~new).sum() * ha)
        if new.any():
            eta_h = float(np.median(e[new]))  # half of the new jam is in place by then
        else:
            eta_h = 0.0
        when = (start + dt.timedelta(hours=eta_h)).astimezone(config.IST)
        p = places[place] if place >= 0 else None
        w = waterways[ww] if ww >= 0 else None
        segments.append({
            "place": p["name"] if p else None,
            "place_ml": p.get("name_ml") if p else None,
            "waterway": w["name"] if w else None,
            "waterway_ml": w.get("name_ml") if w else None,
            "new_jam_area_ha": round(new_ha, 1),
            "already_jammed_area_ha": round(existing_ha, 1),
            "eta_hours": round(eta_h),
            "by_day_en": WEEKDAYS_EN[when.weekday()],
            "by_day_ml": WEEKDAYS_ML[when.weekday()],
            "by_time_ist": when.strftime("%a %d %b %H:00 IST"),
        })

    # Rank by newly blocked area: that is the news. Existing jams follow.
    segments.sort(key=lambda s: (s["new_jam_area_ha"], s["already_jammed_area_ha"]), reverse=True)
    segments = [s for s in segments if s["new_jam_area_ha"] > 0 or s["already_jammed_area_ha"] > 0]
    segments = [s for s in segments if s["place"] or s["waterway"]][: config.TOP_SEGMENTS]

    fs_ist = start.astimezone(config.IST)
    return {
        "synthetic": bool(stats.get("synthetic")),
        "forecast_start_ist": fs_ist.strftime("%a %d %b %Y %H:%M IST"),
        "window_hours": stats["forecast_hours"],
        "barrage": "open" if stats["barrage_open"] else "closed",
        "scene_date": meta.get("scene_date") or wind.get("scene_date"),
        "wind": wind_summary(wind),
        "total_new_jam_area_ha": round(float(sum(s["new_jam_area_ha"] for s in segments)), 1),
        "segments": segments,
        "water_cells": int((cells != 0).sum()),
    }
