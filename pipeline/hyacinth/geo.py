"""Geostationary-satellite conditions over the lake, plus weather context.

Surface solar radiation every 10-15 minutes from geostationary satellites over
the Indian Ocean (EUMETSAT Meteosat IODC / JMA Himawari, via the Open-Meteo
satellite radiation API). Comparing it with a clear-sky model gives:

  cloudiness   1 - measured / clear-sky radiation in daylight (0 clear, 1 overcast)
  insolation   daily solar energy at the surface, kWh/m2 (light and heating that
               drive hyacinth growth and algae)

A few points along the lake are averaged. Rain and temperature are reanalysis /
model data (Open-Meteo, ERA5 + forecast for the last days), labelled as such:
they explain run-off, turbidity and growth, they are not satellite observations.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import urllib.parse
import urllib.request
from collections import defaultdict

from . import config

SAT_URL = "https://satellite-api.open-meteo.com/v1/archive"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
POINTS = [(9.50, 76.37), (9.60, 76.40), (9.70, 76.39)]  # south, centre, north of the lake
UA = {"User-Agent": "hyacinth-forecast/0.1"}
IST = config.IST


def _get(url: str, params: dict) -> object:
    q = urllib.parse.urlencode(params)
    with urllib.request.urlopen(urllib.request.Request(f"{url}?{q}", headers=UA), timeout=120) as r:
        return json.loads(r.read().decode())


def _cos_zenith(t: dt.datetime, lat: float, lon: float) -> float:
    """Solar zenith cosine (NOAA low-precision formulas); t in UTC."""
    doy = t.timetuple().tm_yday
    hour = t.hour + t.minute / 60 + t.second / 3600
    g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                    - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    tst = hour * 60 + eqt + 4 * lon
    ha = math.radians(tst / 4 - 180)
    la = math.radians(lat)
    return math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)


def _clear_sky(cz: float) -> float:
    """Haurwitz clear-sky global horizontal irradiance, W/m2."""
    return 1098.0 * cz * math.exp(-0.057 / cz) if cz > 0.01 else 0.0


def _utc_today() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()


def _sat_series(start: dt.date, end: dt.date, native: bool) -> list[tuple[dt.datetime, float | None, float]]:
    """(UTC mid-interval time, GHI averaged over points or None if missing, clear-sky GHI) per step."""
    end = min(end, _utc_today())
    params = {
        "latitude": ",".join(str(p[0]) for p in POINTS),
        "longitude": ",".join(str(p[1]) for p in POINTS),
        "hourly": "shortwave_radiation",
        "models": "satellite_radiation_seamless",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    if native:
        params["temporal_resolution"] = "native"
    data = _get(SAT_URL, params)
    sites = data if isinstance(data, list) else [data]
    times = sites[0]["hourly"]["time"]
    step = (dt.datetime.fromisoformat(times[1]) - dt.datetime.fromisoformat(times[0])) if len(times) > 1 else dt.timedelta(hours=1)
    out = []
    for i, ts in enumerate(times):
        vals = [s["hourly"]["shortwave_radiation"][i] for s in sites]
        vals = [v for v in vals if v is not None]
        # Values are averages over the preceding interval: evaluate the sun mid-interval.
        t = dt.datetime.fromisoformat(ts).replace(tzinfo=dt.timezone.utc) - step / 2
        cz = sum(_cos_zenith(t, la, lo) for la, lo in POINTS) / len(POINTS)
        out.append((t, sum(vals) / len(vals) if vals else None, _clear_sky(cz)))
    return out


def satellite_daily(start: dt.date, end: dt.date) -> dict[str, dict]:
    """Per IST day: cloudiness (0..1) and insolation (kWh/m2) from geostationary data.

    The feed has outages of a few hours, so cloudiness is measured over the
    daylight steps that exist (measured vs clear-sky on the same steps) and
    insolation is that clear fraction times the full day's clear-sky energy.
    Days with less than half their daylight observed are dropped.
    """
    # day -> [ghi on observed steps, clear-sky on observed steps, clear-sky all steps, n obs, n daylight]
    acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0, 0])
    chunk = dt.timedelta(days=120)
    s = start
    while s <= end:
        e = min(end, s + chunk)
        for t, ghi, cs in _sat_series(s, e, native=False):
            a = acc[t.astimezone(IST).date().isoformat()]
            a[2] += cs
            if cs > 50:
                a[4] += 1
            if ghi is not None:
                a[0] += ghi
                a[1] += cs
                if cs > 50:
                    a[3] += 1
        s = e + dt.timedelta(days=1)
    out = {}
    for day, (ghi, cs_obs, cs_all, n_obs, n_day) in acc.items():
        if n_day == 0 or n_obs < 0.5 * n_day or cs_obs <= 0:
            continue
        clear_frac = min(1.0, max(0.0, ghi / cs_obs))
        out[day] = {"insolation_kwh": round(clear_frac * cs_all / 1000.0, 2), "cloudiness": round(1 - clear_frac, 2)}
    return out


def satellite_now() -> dict | None:
    """Latest native-resolution geostationary reading and today's cloudiness so far."""
    today = dt.datetime.now(dt.timezone.utc).date()
    series = _sat_series(today - dt.timedelta(days=1), today, native=True)
    series = [x for x in series if x[1] is not None]
    if not series:
        return None
    t, ghi, cs = series[-1]
    day_ist = dt.datetime.now(IST).date()
    todays = [(g, c) for tt, g, c in series if tt.astimezone(IST).date() == day_ist and c > 50]
    cloud_today = round(min(1.0, max(0.0, 1 - sum(g for g, _ in todays) / sum(c for _, c in todays))), 2) if todays else None
    last_day = [(g, c) for tt, g, c in series if c > 50]
    cloud_recent = None
    if last_day:
        g, c = last_day[-1]
        cloud_recent = round(min(1.0, max(0.0, 1 - g / c)), 2)
    is_day = cs > 20
    return {
        "time": (t + dt.timedelta(minutes=7, seconds=30)).isoformat().replace("+00:00", "Z"),
        "ghi_wm2": round(ghi),
        "clear_sky_wm2": round(cs),
        "daylight": is_day,
        "cloudiness_latest_daylight": cloud_recent,
        "cloudiness_today": cloud_today,
        "source": "Geostationary satellites (EUMETSAT Meteosat IODC / JMA Himawari) via Open-Meteo",
    }


def weather_daily(start: dt.date, end: dt.date) -> dict[str, dict]:
    """Rain (mm), max temperature (C), max wind (km/h) per IST day: ERA5, then forecast-model recent days."""
    lat, lon = config.WIND_POINT
    daily = "precipitation_sum,temperature_2m_max,wind_speed_10m_max"
    out: dict[str, dict] = {}
    for url, params in (
        # ERA5 lags about five days; the forecast models fill the rest.
        (ARCHIVE_URL, {"start_date": start.isoformat(),
                       "end_date": min(end, _utc_today() - dt.timedelta(days=6)).isoformat()}),
        (FORECAST_URL, {"past_days": 14, "forecast_days": 1}),
    ):
        try:
            d = _get(url, {"latitude": lat, "longitude": lon, "daily": daily, "timezone": "Asia/Kolkata", **params})["daily"]
        except Exception as e:  # weather is context; never fail the dataset on it
            print(f"[geo] weather from {url} failed: {e}")
            continue
        for i, day in enumerate(d["time"]):
            if d["precipitation_sum"][i] is None:
                continue
            if day > end.isoformat() or day < start.isoformat():
                continue
            if url == FORECAST_URL and day in out:
                continue  # prefer the reanalysis where both exist
            out[day] = {
                "rain_mm": round(d["precipitation_sum"][i], 1),
                "tmax_c": d["temperature_2m_max"][i],
                "wind_max_kmh": d["wind_speed_10m_max"][i],
            }
    return out
