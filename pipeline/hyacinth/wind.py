"""Wind forcing from Open-Meteo (free, no key).

* Forecast API with ``past_days``: covers the spin-up from the satellite scene to
  "now" with analysed wind, then the 72 h forecast.
* Archive API (ERA5): historical wind for hindcasts.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import requests

from . import config

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
UA = {"User-Agent": "hyacinth-forecast/0.1"}


def _floor_hour(t: dt.datetime) -> dt.datetime:
    return t.astimezone(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)


def _iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _uv(speed: np.ndarray, dir_from: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.deg2rad(dir_from)
    return -speed * np.sin(r), -speed * np.cos(r)


def _fill_gaps(a: list) -> np.ndarray:
    x = np.array([np.nan if v is None else float(v) for v in a])
    ok = ~np.isnan(x)
    if not ok.any():
        raise RuntimeError("Open-Meteo returned no wind values")
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def _hourly(url: str, params: dict) -> tuple[list[dt.datetime], np.ndarray, np.ndarray]:
    lat, lon = config.WIND_POINT
    q = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms",
        "timezone": "GMT",
        **params,
    }
    r = requests.get(url, params=q, headers=UA, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    times = [dt.datetime.fromisoformat(t).replace(tzinfo=dt.timezone.utc) for t in h["time"]]
    speed = _fill_gaps(h["wind_speed_10m"])
    # Average directions as vectors, never as angles, so fill gaps on u/v.
    d = h["wind_direction_10m"]
    if any(v is None for v in d):
        s_raw = np.array([np.nan if v is None else v for v in h["wind_speed_10m"]], float)
        d_raw = np.array([np.nan if v is None else v for v in d], float)
        u, v = _uv(np.nan_to_num(s_raw), np.nan_to_num(d_raw))
        bad = np.isnan(d_raw) | np.isnan(s_raw)
        u[bad] = np.nan
        v[bad] = np.nan
        u = _fill_gaps(list(np.where(np.isnan(u), None, u)))
        v = _fill_gaps(list(np.where(np.isnan(v), None, v)))
        return times, u, v
    u, v = _uv(speed, np.array(d, float))
    return times, u, v


def _slice(times, u, v, start: dt.datetime, end: dt.datetime):
    sel = [i for i, t in enumerate(times) if start <= t <= end]
    if not sel or times[sel[0]] != start:
        raise RuntimeError(f"wind record does not cover {start.isoformat()} .. {end.isoformat()}")
    expected = int((end - start).total_seconds() // 3600) + 1
    if len(sel) < expected:
        raise RuntimeError(f"wind record has {len(sel)} of {expected} hours")
    return u[sel], v[sel]


def forcing(
    start: dt.datetime,
    forecast_start: dt.datetime,
    forecast_hours: int,
    *,
    historical: bool = False,
    barrage_open: bool,
    scene_date: str | None,
) -> dict:
    """Build wind.json covering ``start`` .. ``forecast_start + forecast_hours``."""
    start = _floor_hour(start)
    forecast_start = _floor_hour(forecast_start)
    end = forecast_start + dt.timedelta(hours=forecast_hours)
    if forecast_start < start:
        raise ValueError("forecast_start is before start")

    if historical:
        times, u, v = _hourly(
            ARCHIVE_URL,
            {"start_date": start.date().isoformat(), "end_date": end.date().isoformat()},
        )
        source = "Open-Meteo archive (ERA5)"
    else:
        now = dt.datetime.now(dt.timezone.utc)
        past_days = max(0, math.ceil((now - start).total_seconds() / 86400)) + 1
        if past_days > 92:
            raise RuntimeError("scene is older than Open-Meteo's 92 past days; pick a newer scene")
        fdays = max(1, math.ceil((end - now).total_seconds() / 86400) + 1)
        times, u, v = _hourly(FORECAST_URL, {"past_days": past_days, "forecast_days": min(16, fdays)})
        source = "Open-Meteo forecast"

    u, v = _slice(times, u, v, start, end)
    return {
        "start": _iso(start),
        "forecast_start": _iso(forecast_start),
        "forecast_hours": forecast_hours,
        "step_hours": 1,
        "u": [round(float(x), 3) for x in u],
        "v": [round(float(x), 3) for x in v],
        "dx_m": config.DX_M,
        "barrage_open": barrage_open,
        "source": source,
        "scene_date": scene_date,
        "synthetic": False,
    }


def summary(w: dict) -> dict:
    """Wind summary over the forecast window, for captions."""
    lead = int(round((dt.datetime.fromisoformat(w["forecast_start"].replace("Z", "+00:00"))
                      - dt.datetime.fromisoformat(w["start"].replace("Z", "+00:00"))).total_seconds() / 3600))
    u = np.array(w["u"][lead:lead + w["forecast_hours"] + 1])
    v = np.array(w["v"][lead:lead + w["forecast_hours"] + 1])
    speed = np.hypot(u, v)
    mu, mv = u.mean(), v.mean()
    dir_from = (math.degrees(math.atan2(-mu, -mv)) + 360) % 360
    names = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
    return {
        "mean_speed_ms": round(float(speed.mean()), 1),
        "max_speed_ms": round(float(speed.max()), 1),
        "prevailing_from_deg": round(dir_from),
        "prevailing_from": names[int((dir_from + 22.5) // 45) % 8],
    }
