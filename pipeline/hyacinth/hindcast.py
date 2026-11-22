"""Hindcast scoring: does the sim beat "mats stay where they were"?

Compares the simulated mat field at the end of the run with the observed
day-N scene, inside water cells that were cloud-free on both days, against the
persistence baseline (the day-0 seed itself).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from .dataset import read_bin
from .io import INFLOW, WATER, read_grey, read_json, read_mask, write_json


def _f1(pred: np.ndarray, obs: np.ndarray, valid: np.ndarray, tol: int) -> dict:
    p = pred & valid
    o = obs & valid
    if tol > 0:
        st = ndimage.generate_binary_structure(2, 2)
        o_near = ndimage.binary_dilation(o, st, iterations=tol)
        p_near = ndimage.binary_dilation(p, st, iterations=tol)
        tp_p = int((p & o_near).sum())  # predicted cells near an observed mat
        tp_o = int((o & p_near).sum())  # observed cells near a predicted mat
    else:
        tp_p = tp_o = int((p & o).sum())
    precision = tp_p / max(1, int(p.sum()))
    recall = tp_o / max(1, int(o.sum()))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def score(d: Path) -> dict:
    stats = read_json(d / "out" / "stats.json")
    shape = (stats["height"], stats["width"])
    cells = read_mask(d / "mask.png")
    water = (cells == WATER) | (cells == INFLOW)
    seed = read_grey(d / "seed.png") > 0
    obs = read_grey(d / "observed.png") > 0
    clear = np.load(d / "hindcast_clear.npy") if (d / "hindcast_clear.npy").exists() else np.ones(shape, bool)
    valid = water & clear
    density = read_bin(d / "out" / "density.bin", shape)
    pred = density >= 0.25 * stats["particles_per_full_cell"]

    result = {"valid_cells": int(valid.sum()), "observed_mat_cells": int((obs & valid).sum())}
    for tol in (0, 2, 5):
        key = f"tolerance_{tol}_cells"
        result[key] = {"model": _f1(pred, obs, valid, tol), "persistence": _f1(seed, obs, valid, tol)}
    result["model_beats_persistence_at_2_cells"] = (
        result["tolerance_2_cells"]["model"]["f1"] > result["tolerance_2_cells"]["persistence"]["f1"]
    )
    write_json(result, d / "out" / "hindcast_score.json")
    return result
