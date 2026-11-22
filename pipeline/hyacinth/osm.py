"""Lake and canal mask from OpenStreetMap (Overpass API), plus name rasters for
captions: which waterway and which nearby place each cell belongs to.

Output:
  cells   uint8 [H, W]  LAND / WATER / INFLOW / OUTLET / BARRAGE
  labels  dict with 'waterway_id' int16 [H, W], 'place_id' int16 [H, W],
          'waterways' [{name, name_ml}], 'places' [{name, name_ml, lon, lat}]
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
import requests
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from scipy import ndimage
from shapely.geometry import LineString, Polygon, mapping
from shapely.geometry import shape as to_shape
from shapely.ops import polygonize, unary_union

from . import config
from .grid import Grid
from .io import BARRAGE, INFLOW, LAND, OUTLET, WATER

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def _query(bbox) -> dict:
    lon0, lat0, lon1, lat1 = bbox
    b = f"{lat0},{lon0},{lat1},{lon1}"
    q = f"""
[out:json][timeout:300];
(
  way["natural"="water"]({b});
  relation["natural"="water"]({b});
  way["waterway"="riverbank"]({b});
  relation["waterway"="riverbank"]({b});
  way["waterway"~"^(river|canal)$"]({b});
  node["place"~"^(town|village|suburb|hamlet|island|islet)$"]({b});
  way["name"~"{config.BARRAGE_OSM_NAME}",i]["waterway"~"^(dam|weir)$"]({b});
  way["name"~"{config.BARRAGE_OSM_NAME}",i]["man_made"~"^(dyke|bridge)$"]({b});
);
out geom;
"""
    last = None
    for url in OVERPASS_URLS:
        for attempt in range(3):
            try:
                r = requests.post(url, data={"data": q}, timeout=400,
                                  headers={"User-Agent": "hyacinth-forecast/0.1"})
                if r.status_code == 200:
                    return r.json()
                last = RuntimeError(f"{url}: HTTP {r.status_code}")
            except requests.RequestException as e:  # network hiccup: retry
                last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Overpass query failed: {last}")


def _line(geom) -> LineString | None:
    pts = [(p["lon"], p["lat"]) for p in geom or [] if p]
    return LineString(pts) if len(pts) >= 2 else None


def _way_polygon(el) -> Polygon | None:
    pts = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
    if len(pts) >= 4 and pts[0] == pts[-1]:
        poly = Polygon(pts)
        return poly if poly.is_valid else poly.buffer(0)
    return None


def _relation_polygon(el):
    outers, inners = [], []
    for m in el.get("members", []):
        if m.get("type") != "way":
            continue
        line = _line(m.get("geometry"))
        if line is None:
            continue
        (inners if m.get("role") == "inner" else outers).append(line)
    outer = unary_union(list(polygonize(unary_union(outers)))) if outers else None
    if outer is None or outer.is_empty:
        return None
    if inners:
        inner = unary_union(list(polygonize(unary_union(inners))))
        outer = outer.difference(inner)
    return outer


def _project(geom, grid: Grid):
    return transform_geom("EPSG:4326", grid.crs, mapping(geom))


def build(grid: Grid, bbox=config.BBOX):
    t0 = time.time()
    osm = _query(bbox)
    print(f"[osm] {len(osm.get('elements', []))} elements in {time.time() - t0:.0f}s")

    polys, lines, barrage_geoms = [], [], []
    places = []
    waterways: dict[str, dict] = {}
    named_shapes: list[tuple[dict, str]] = []  # (projected geom, name)

    for el in osm["elements"]:
        tags = el.get("tags", {})
        name = tags.get("name")
        if el["type"] == "node" and "place" in tags and name:
            places.append({"name": name, "name_ml": tags.get("name:ml"), "lon": el["lon"], "lat": el["lat"]})
            continue
        if config.BARRAGE_OSM_NAME.lower() in (name or "").lower() and (
            tags.get("waterway") in ("dam", "weir") or tags.get("man_made") in ("dyke", "bridge")
        ):
            line = _line(el.get("geometry"))
            if line is not None:
                barrage_geoms.append(line)
            continue
        ww = tags.get("waterway")
        if el["type"] == "way" and ww in config.WATERWAY_WIDTH_M:
            line = _line(el.get("geometry"))
            if line is None:
                continue
            lines.append((line, ww))
            if name:
                waterways.setdefault(name, {"name": name, "name_ml": tags.get("name:ml")})
                named_shapes.append((_project(line, grid), name, ww))
            continue
        poly = _way_polygon(el) if el["type"] == "way" else _relation_polygon(el)
        if poly is None or poly.is_empty:
            continue
        polys.append(poly)
        if name:
            waterways.setdefault(name, {"name": name, "name_ml": tags.get("name:ml")})
            named_shapes.append((_project(poly, grid), name, "area"))

    if not polys and not lines:
        raise RuntimeError("OSM returned no water in the bounding box")

    shape = grid.shape
    water = np.zeros(shape, bool)
    if polys:
        water |= rasterize(((_project(p, grid), 1) for p in polys), out_shape=shape,
                           transform=grid.transform, fill=0, dtype="uint8").astype(bool)
    min_w = config.MIN_CHANNEL_CELLS * grid.dx
    line_shapes = []
    for line, ww in lines:
        g = _project(line, grid)
        width = max(config.WATERWAY_WIDTH_M[ww], min_w)
        line_shapes.append((mapping(to_shape(g).buffer(width / 2)), 1))
    if line_shapes:
        water |= rasterize(line_shapes, out_shape=shape, transform=grid.transform, fill=0,
                           all_touched=True, dtype="uint8").astype(bool)

    # Keep only the water body connected to the main lake (drop isolated ponds).
    lab, n = ndimage.label(water)
    if n == 0:
        raise RuntimeError("water mask is empty after rasterisation")
    sizes = ndimage.sum(water, lab, index=np.arange(1, n + 1))
    water = lab == (1 + int(np.argmax(sizes)))

    cells = np.where(water, WATER, LAND).astype(np.uint8)
    water_fine = _fine_water(grid, polys, lines, water)

    # Barrage.
    if barrage_geoms:
        print(f"[osm] barrage from OSM ({len(barrage_geoms)} feature(s))")
        bar_shapes = [mapping(to_shape(_project(g, grid)).buffer(config.BARRAGE_WIDTH_M / 2))
                      for g in barrage_geoms]
    else:
        print("[osm] barrage not found in OSM; using configured fallback line")
        line = LineString(config.BARRAGE_FALLBACK_LINE)
        bar_shapes = [mapping(to_shape(_project(line, grid)).buffer(config.BARRAGE_WIDTH_M / 2))]
    bar = rasterize(((s, 1) for s in bar_shapes), out_shape=shape, transform=grid.transform,
                    fill=0, all_touched=True, dtype="uint8").astype(bool)
    cells[bar & water] = BARRAGE

    # Open boundaries: water on the north edge is the lake continuing to Kochi
    # and the sea (outlet); water on the east and south edges is a river
    # entering the domain (inflow).
    edge_n = np.zeros(shape, bool)
    edge_n[0, :] = True
    edge_es = np.zeros(shape, bool)
    edge_es[:, -1] = True
    edge_es[-1, :] = True
    cells[edge_n & (cells == WATER)] = OUTLET
    cells[edge_es & (cells == WATER)] = INFLOW

    check_barrage_seals(cells)

    labels = _labels(grid, cells, named_shapes, waterways, places)
    return cells, labels, water_fine


def _fine_water(grid: Grid, polys, lines, coarse_water: np.ndarray) -> np.ndarray:
    """Water at the Sentinel-2 10 m resolution, with true channel widths, used
    for mat detection only (the simulation grid widens narrow canals)."""
    k = config.FINE_FACTOR
    fine = grid.subdivide(k)
    wf = np.zeros(fine.shape, bool)
    if polys:
        wf |= rasterize(((_project(p, grid), 1) for p in polys), out_shape=fine.shape,
                        transform=fine.transform, fill=0, dtype="uint8").astype(bool)
    shapes = [(mapping(to_shape(_project(line, grid)).buffer(config.WATERWAY_WIDTH_M[ww] / 2)), 1)
              for line, ww in lines]
    if shapes:
        wf |= rasterize(shapes, out_shape=fine.shape, transform=fine.transform, fill=0,
                        dtype="uint8").astype(bool)
    return wf & np.repeat(np.repeat(coarse_water, k, axis=0), k, axis=1)


def _reachable(cells: np.ndarray, passable: np.ndarray) -> np.ndarray:
    """8-connected flood fill from the sea outlet (D2Q9 also streams diagonally)."""
    h, w = cells.shape
    seen = np.zeros_like(passable)
    q = deque(zip(*np.nonzero(cells == OUTLET)))
    for y, x in q:
        seen[y, x] = True
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and passable[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return seen


def check_barrage_seals(cells: np.ndarray) -> dict:
    """Does the barrage line close the open lake?

    Two questions, answered separately:
      1. Lake: with the barrage closed, is the wide water south of it still
         connected to the wide water north of it? Narrow canals (< ~7 cells)
         are removed first. This must be "no", or the barrage line is wrong.
      2. Canals: is any water south of the barrage still connected to the sea
         through narrow canals around the bund? That is real geography
         (Kuttanad canals have their own regulators, not modelled), so it is
         reported, not treated as an error.
    """
    water = (cells == WATER) | (cells == INFLOW) | (cells == OUTLET)
    report = {"has_outlet": bool((cells == OUTLET).any()), "has_barrage": bool((cells == BARRAGE).any())}
    if not report["has_outlet"]:
        print("[osm] WARNING: no outlet cells on the north edge")
        return report
    if not report["has_barrage"]:
        print("[osm] WARNING: no barrage cells in the water mask")
        return report

    wide = ndimage.binary_opening(water, structure=np.ones((3, 3), bool), iterations=3)
    lab, _ = ndimage.label(wide, structure=np.ones((3, 3), bool))
    near = ndimage.binary_dilation(cells == BARRAGE, structure=np.ones((3, 3), bool), iterations=5)
    ids = set(np.unique(lab[near & wide]).tolist()) - {0}
    report["lake_sealed"] = len(ids) >= 2

    closed = _reachable(cells, water)
    opened = _reachable(cells, water | (cells == BARRAGE))
    behind = opened & ~closed & (cells != BARRAGE)
    report["cells_cut_off_by_barrage"] = int(behind.sum())
    total = int(water.sum())
    if not report["lake_sealed"]:
        print("[osm] WARNING: the barrage line does not close the open lake; "
              "fix BARRAGE_FALLBACK_LINE / the OSM feature in config.py")
    elif behind.sum() == 0:
        print("[osm] barrage closes the open lake, but narrow canals connect every part of the network "
              "to the sea around it; closing the barrage only blocks the lake itself")
    else:
        print(f"[osm] barrage closes the open lake: {int(behind.sum())} of {total} water cells cut off from the sea")
    return report


def _labels(grid: Grid, cells, named_shapes, waterways, places) -> dict:
    shape = grid.shape
    names = list(waterways)
    idx = {n: i for i, n in enumerate(names)}
    # Burn areas first, then rivers, then canals, so the narrowest feature wins.
    order = {"area": 0, "river": 1, "canal": 2}
    ww_id = np.full(shape, -1, np.int16)
    for g, name, kind in sorted(named_shapes, key=lambda s: order[s[2]]):
        geom = to_shape(g)
        if kind != "area":
            geom = geom.buffer(max(config.WATERWAY_WIDTH_M[kind], config.MIN_CHANNEL_CELLS * grid.dx))
        burn = rasterize([(mapping(geom), 1)], out_shape=shape, transform=grid.transform,
                         fill=0, all_touched=True, dtype="uint8").astype(bool)
        ww_id[burn] = idx[name]
    ww_id[cells == LAND] = -1

    # Nearest place for every water cell.
    place_id = np.full(shape, -1, np.int16)
    if places:
        pc = np.array([grid.lonlat_to_cell(p["lon"], p["lat"]) for p in places])  # (col, row)
        ys, xs = np.nonzero(cells != LAND)
        best = np.full(len(xs), np.inf)
        arg = np.zeros(len(xs), np.int16)
        for k, (pcx, pcy) in enumerate(pc):
            d = (xs + 0.5 - pcx) ** 2 + (ys + 0.5 - pcy) ** 2
            better = d < best
            best[better] = d[better]
            arg[better] = k
        place_id[ys, xs] = arg
    return {
        "waterway_id": ww_id,
        "place_id": place_id,
        "waterways": [waterways[n] for n in names],
        "places": places,
    }
