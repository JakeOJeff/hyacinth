"""A dataset directory: the three contract files plus pipeline-side extras.

  mask.png, seed.png, wind.json      -> read by the simulation
  observed.png                       -> optional, hindcast truth (day N mats)
  labels.npz, labels.json, grid.json -> pipeline only (caption names)
  meta.json                          -> provenance for the web page / captions
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .io import read_json, write_grey, write_json, write_mask


def save(
    out: Path,
    *,
    cells: np.ndarray,
    seed: np.ndarray,
    wind: dict,
    labels: dict,
    grid: dict,
    meta: dict,
    observed: np.ndarray | None = None,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if cells.shape != seed.shape:
        raise ValueError("mask and seed differ in shape")
    write_mask(cells, out / "mask.png")
    write_grey(seed, out / "seed.png")
    write_json(wind, out / "wind.json")
    if observed is not None:
        write_grey(observed, out / "observed.png")
    elif (out / "observed.png").exists():
        (out / "observed.png").unlink()
    np.savez_compressed(out / "labels.npz", waterway_id=labels["waterway_id"], place_id=labels["place_id"])
    write_json({"waterways": labels["waterways"], "places": labels["places"]}, out / "labels.json")
    write_json(grid, out / "grid.json")
    write_json(meta, out / "meta.json")
    # Stale render output must never be paired with new inputs.
    if (out / "out").exists():
        shutil.rmtree(out / "out")


def load_labels(d: Path) -> dict:
    z = np.load(d / "labels.npz")
    names = read_json(d / "labels.json")
    return {"waterway_id": z["waterway_id"], "place_id": z["place_id"], **names}


def read_bin(path: Path, shape: tuple[int, int]) -> np.ndarray:
    a = np.fromfile(path, dtype="<f4")
    if a.size != shape[0] * shape[1]:
        raise ValueError(f"{path.name}: {a.size} values, expected {shape[0] * shape[1]}")
    return a.reshape(shape)
