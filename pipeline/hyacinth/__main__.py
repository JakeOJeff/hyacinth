"""Hyacinth command line.

  python -m hyacinth synth                      offline synthetic dataset -> data/synthetic
  python -m hyacinth fetch                      today's real inputs       -> data/latest
  python -m hyacinth hindcast-dates 2025-12-01 2026-03-31
  python -m hyacinth hindcast DAY0 DAY5         hindcast inputs           -> data/hindcast
  python -m hyacinth render DIR                 run the GPU sim (Deno)
  python -m hyacinth caption DIR [--no-model]   facts + Malayalam/English caption
  python -m hyacinth score DIR                  hindcast score vs persistence
  python -m hyacinth publish DIR NAME           copy to web/public/data/NAME for the site
  python -m hyacinth post DIR                   post to Telegram
  python -m hyacinth health [--start D] [--end D] [--limit N]
                                                 add new Sentinel-2 passes to the lake-health dataset
  python -m hyacinth health-publish             health.json + maps -> web/public/data/health
  python -m hyacinth daily                      fetch, render, caption, publish, health, post
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import config
from .io import read_json, write_json

STATIC = config.DATA_DIR / "static"


def _base_mask(refresh: bool):
    """OSM mask + labels + 10 m water, cached: OSM changes slowly and Overpass is shared."""
    from . import osm
    from .grid import Grid

    grid = Grid.from_bbox()
    cache = STATIC / "osm_base.npz"
    if cache.exists() and not refresh:
        z = np.load(cache, allow_pickle=False)
        names = read_json(STATIC / "osm_labels.json")
        g = read_json(STATIC / "grid.json")
        if g == grid.to_json() and "water_fine" in z.files:
            labels = {"waterway_id": z["waterway_id"], "place_id": z["place_id"], **names}
            k = config.FINE_FACTOR
            water_fine = np.unpackbits(z["water_fine"], count=grid.width * k * grid.height * k)
            water_fine = water_fine.reshape(grid.height * k, grid.width * k).astype(bool)
            print(f"[osm] using cached mask {grid.width}x{grid.height}")
            osm.check_barrage_seals(z["cells"])
            return grid, z["cells"], labels, water_fine
        print("[osm] cache is stale; rebuilding")
    cells, labels, water_fine = osm.build(grid)
    STATIC.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, cells=cells, waterway_id=labels["waterway_id"], place_id=labels["place_id"],
                        water_fine=np.packbits(water_fine.ravel()))
    write_json({"waterways": labels["waterways"], "places": labels["places"]}, STATIC / "osm_labels.json")
    write_json(grid.to_json(), STATIC / "grid.json")
    from .io import write_mask
    write_mask(cells, STATIC / "mask_preview.png")
    return grid, cells, labels, water_fine


def cmd_synth(a):
    from . import dataset, synth
    from .osm import check_barrage_seals

    start = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    cells, seed, wind, labels, grid = synth.build(start)
    check_barrage_seals(cells)
    meta = {"source": "synthetic", "scene_date": None, "generated": start.isoformat(), "synthetic": True}
    dataset.save(Path(a.out), cells=cells, seed=seed, wind=wind, labels=labels, grid=grid, meta=meta)
    print(f"[synth] wrote {a.out} ({cells.shape[1]}x{cells.shape[0]}, {int((seed > 0).sum())} mat cells)")


def cmd_fetch(a):
    from . import dataset, sentinel, wind

    now = dt.datetime.now(dt.timezone.utc)
    until = dt.date.fromisoformat(a.date) if a.date else now.date()
    grid, cells, labels, water_fine = _base_mask(a.refresh_osm)
    scene = sentinel.latest_scene(grid, cells, water_fine, until)
    age_h = (now - scene.when).total_seconds() / 3600
    print(f"[fetch] scene {scene.when:%Y-%m-%d %H:%M} UTC ({age_h:.0f} h old), tiles {scene.tiles}")
    forcing = wind.forcing(scene.when, now, a.hours, barrage_open=config.barrage_open(now.date()),
                           scene_date=scene.when.date().isoformat())
    meta = {
        "source": "sentinel-2-l2a + osm + open-meteo",
        "scene_date": scene.when.date().isoformat(),
        "scene_tiles": scene.tiles,
        "clear_water": round(scene.clear_water, 3),
        "mat_cells": scene.mat_cells,
        "generated": now.isoformat(),
        "synthetic": False,
    }
    dataset.save(Path(a.out), cells=cells, seed=scene.seed, wind=forcing, labels=labels,
                 grid=grid.to_json(), meta=meta)
    print(f"[fetch] wrote {a.out}")


def cmd_hindcast_dates(a):
    from . import sentinel

    for d, cc in sentinel.clear_dates(dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end), a.max_cloud):
        print(f"{d}  tile cloud {cc:5.1f}%")


def cmd_hindcast(a):
    from . import dataset, sentinel, wind

    d0, d1 = dt.date.fromisoformat(a.day0), dt.date.fromisoformat(a.day1)
    if d1 <= d0:
        sys.exit("day1 must be after day0")
    grid, cells, labels, water_fine = _base_mask(a.refresh_osm)
    s0 = sentinel.scene_on(grid, cells, water_fine, d0)
    s1 = sentinel.scene_on(grid, cells, water_fine, d1)
    print(f"[hindcast] day0 clear {s0.clear_water:.0%} mats {s0.mat_cells}; "
          f"day{(d1 - d0).days} clear {s1.clear_water:.0%} mats {s1.mat_cells}")
    hours = int(round((s1.when - s0.when).total_seconds() / 3600))
    forcing = wind.forcing(s0.when, s0.when, hours, historical=True,
                           barrage_open=config.barrage_open(d0), scene_date=d0.isoformat())
    meta = {
        "source": "hindcast: sentinel-2-l2a + osm + open-meteo archive",
        "scene_date": d0.isoformat(),
        "observed_date": d1.isoformat(),
        "clear_water": [round(s0.clear_water, 3), round(s1.clear_water, 3)],
        "synthetic": False,
    }
    out = Path(a.out)
    dataset.save(out, cells=cells, seed=s0.seed, wind=forcing, labels=labels, grid=grid.to_json(),
                 meta=meta, observed=s1.seed)
    np.save(out / "hindcast_clear.npy", s0.clear & s1.clear)
    print(f"[hindcast] wrote {out}: {hours} h from {s0.when:%Y-%m-%d %H:%M} to {s1.when:%Y-%m-%d %H:%M} UTC")


def run_render(d: Path, extra: list[str] | None = None) -> None:
    deno = shutil.which("deno")
    if not deno:
        sys.exit("deno not found on PATH (npm i -g deno, or https://deno.land)")
    cmd = [deno, "task", "render", "--data", str(d.resolve()), *(extra or [])]
    print("[render] " + " ".join(cmd))
    subprocess.run(cmd, cwd=config.ROOT, check=True)


def cmd_render(a):
    run_render(Path(a.dir), a.extra)


def cmd_caption(a):
    from .aggregate import jam_facts
    from .caption import make_caption

    d = Path(a.dir)
    facts = jam_facts(d)
    write_json(facts, d / "out" / "facts.json")
    cap = make_caption(facts, use_model=not a.no_model)
    write_json(cap, d / "out" / "caption.json")
    print(cap["text"])


def cmd_score(a):
    import json
    from .hindcast import score

    print(json.dumps(score(Path(a.dir)), indent=2))


PUBLISH_FILES = ["mask.png", "seed.png", "wind.json", "observed.png", "meta.json"]
PUBLISH_OUT = ["forecast.mp4", "heatmap.png", "final.png", "caption.json", "facts.json",
               "stats.json", "hindcast_score.json", "state.bin"]


def cmd_publish(a):
    src = Path(a.dir)
    dst = config.WEB_DATA_DIR / a.name
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in PUBLISH_FILES:
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    for f in PUBLISH_OUT:
        if (src / "out" / f).exists():
            shutil.copy2(src / "out" / f, dst / f)
    _publish_places(src, dst)
    _write_index()
    print(f"[publish] {src} -> {dst}")


def _publish_places(src: Path, dst: Path) -> None:
    """places.png (16-bit grey, OSM place id + 1 per cell, 0 = none) and
    places.json (names), so the site can name places in what-if runs."""
    if not ((src / "labels.npz").exists() and (src / "labels.json").exists()):
        return
    import numpy as np
    from PIL import Image
    from .dataset import load_labels

    labels = load_labels(src)
    ids = (labels["place_id"].astype(np.int32) + 1).clip(0, 65535).astype(np.uint16)
    Image.fromarray(ids).save(dst / "places.png")
    write_json([{"name": p["name"], "name_ml": p.get("name_ml")} for p in labels["places"]], dst / "places.json")


def _write_index():
    """web/public/data/index.json: the datasets the site can switch between."""
    entries = []
    for d in sorted(p for p in config.WEB_DATA_DIR.iterdir() if p.is_dir()):
        if not (d / "wind.json").exists():
            continue
        meta = read_json(d / "meta.json") if (d / "meta.json").exists() else {}
        entries.append({
            "name": d.name,
            "source": meta.get("source"),
            "scene_date": meta.get("scene_date"),
            "observed_date": meta.get("observed_date"),
            "synthetic": bool(meta.get("synthetic")),
        })
    write_json(entries, config.WEB_DATA_DIR / "index.json")


def cmd_post(a):
    from .telegram import post

    d = Path(a.dir)
    cap = read_json(d / "out" / "caption.json")
    post(d / "out", cap["text"])


def cmd_daily(a):
    d = config.DATA_DIR / "latest"
    cmd_fetch(argparse.Namespace(out=str(d), date=None, hours=72, refresh_osm=a.refresh_osm))
    run_render(d)
    cmd_caption(argparse.Namespace(dir=str(d), no_model=a.no_model))
    cmd_publish(argparse.Namespace(dir=str(d), name="latest"))
    try:
        cmd_health(argparse.Namespace(start=None, end=None, limit=None))
        cmd_health_publish(argparse.Namespace())
    except Exception as e:  # the forecast must still go out if the health update fails
        print(f"[daily] lake-health update failed: {type(e).__name__}: {e}")
    if not a.no_post:
        cmd_post(argparse.Namespace(dir=str(d)))


def cmd_health(a):
    from .health import HEALTH_START, build

    start = dt.date.fromisoformat(a.start) if a.start else HEALTH_START
    end = dt.date.fromisoformat(a.end) if a.end else None
    print(f"[health] added {build(start, end, a.limit)} pass(es)")


def cmd_health_publish(a):
    from .healthpub import publish

    publish()


def _load_dotenv() -> None:
    """KEY=value lines from the repo-root .env.local and .env (first wins), never overriding real env vars."""
    import os
    for path in (config.ROOT / ".env.local", config.ROOT / ".env"):
        if path.exists():
            _read_env(path, os)


def _read_env(path, os) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


def main(argv=None):
    _load_dotenv()
    p = argparse.ArgumentParser(prog="hyacinth", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth")
    s.add_argument("--out", default=str(config.DATA_DIR / "synthetic"))
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("fetch")
    s.add_argument("--out", default=str(config.DATA_DIR / "latest"))
    s.add_argument("--date", help="search for scenes up to this date (default today)")
    s.add_argument("--hours", type=int, default=72)
    s.add_argument("--refresh-osm", action="store_true")
    s.set_defaults(fn=cmd_fetch)

    s = sub.add_parser("hindcast-dates")
    s.add_argument("start")
    s.add_argument("end")
    s.add_argument("--max-cloud", type=float, default=20.0)
    s.set_defaults(fn=cmd_hindcast_dates)

    s = sub.add_parser("hindcast")
    s.add_argument("day0")
    s.add_argument("day1")
    s.add_argument("--out", default=str(config.DATA_DIR / "hindcast"))
    s.add_argument("--refresh-osm", action="store_true")
    s.set_defaults(fn=cmd_hindcast)

    s = sub.add_parser("render")
    s.add_argument("dir")
    s.add_argument("extra", nargs=argparse.REMAINDER, help="passed to render.ts (e.g. --fps 24)")
    s.set_defaults(fn=cmd_render)

    s = sub.add_parser("caption")
    s.add_argument("dir")
    s.add_argument("--no-model", action="store_true", help="template caption, no model call")
    s.set_defaults(fn=cmd_caption)

    s = sub.add_parser("score")
    s.add_argument("dir")
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("publish")
    s.add_argument("dir")
    s.add_argument("name")
    s.set_defaults(fn=cmd_publish)

    s = sub.add_parser("post")
    s.add_argument("dir")
    s.set_defaults(fn=cmd_post)

    s = sub.add_parser("health")
    s.add_argument("--start", help="first date (default 2025-01-01)")
    s.add_argument("--end", help="last date (default today)")
    s.add_argument("--limit", type=int, help="process at most N new dates")
    s.set_defaults(fn=cmd_health)

    s = sub.add_parser("health-publish")
    s.set_defaults(fn=cmd_health_publish)

    s = sub.add_parser("daily")
    s.add_argument("--no-model", action="store_true")
    s.add_argument("--no-post", action="store_true")
    s.add_argument("--refresh-osm", action="store_true")
    s.set_defaults(fn=cmd_daily)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
