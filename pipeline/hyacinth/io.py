"""Writers / readers for the three-file contract (mask.png, seed.png, wind.json)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

LAND, WATER, INFLOW, OUTLET, BARRAGE = 0, 1, 2, 3, 4

PALETTE = {
    LAND: (0, 0, 0),
    WATER: (255, 255, 255),
    INFLOW: (0, 255, 0),
    OUTLET: (0, 0, 255),
    BARRAGE: (255, 0, 0),
}


def write_mask(cells: np.ndarray, path: Path) -> None:
    rgb = np.zeros(cells.shape + (3,), np.uint8)
    for t, colour in PALETTE.items():
        rgb[cells == t] = colour
    Image.fromarray(rgb, "RGB").save(path, optimize=True)


def read_mask(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.int32)
    best = np.zeros(rgb.shape[:2], np.int32)
    best_d = np.full(rgb.shape[:2], np.iinfo(np.int32).max)
    for t, colour in PALETTE.items():
        d = ((rgb - np.array(colour)) ** 2).sum(-1)
        better = d < best_d
        best[better] = t
        best_d[better] = d[better]
    return best.astype(np.uint8)


def write_grey(values: np.ndarray, path: Path) -> None:
    Image.fromarray(np.clip(values, 0, 255).astype(np.uint8), "L").save(path, optimize=True)


def read_grey(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def write_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
