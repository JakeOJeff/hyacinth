"""The simulation grid: a north-up UTM raster, one pixel per LBM cell."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.warp import transform as warp_transform

from . import config


@dataclass(frozen=True)
class Grid:
    crs: str
    transform: Affine
    width: int
    height: int
    dx: float

    @classmethod
    def from_bbox(cls, bbox=config.BBOX, crs=config.GRID_CRS, dx=config.DX_M) -> "Grid":
        lon0, lat0, lon1, lat1 = bbox
        lons = [lon0, lon1, lon1, lon0]
        lats = [lat0, lat0, lat1, lat1]
        xs, ys = warp_transform("EPSG:4326", crs, lons, lats)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        # Snap the origin to the cell size so runs are reproducible.
        xmin = math.floor(xmin / dx) * dx
        ymax = math.ceil(ymax / dx) * dx
        width = int(math.ceil((xmax - xmin) / dx))
        height = int(math.ceil((ymax - ymin) / dx))
        return cls(crs, Affine(dx, 0.0, xmin, 0.0, -dx, ymax), width, height, dx)

    def subdivide(self, k: int) -> "Grid":
        """Same footprint, k x k sub-cells per cell."""
        return Grid(self.crs, self.transform * Affine.scale(1 / k), self.width * k, self.height * k, self.dx / k)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def lonlat_to_cell(self, lon: float, lat: float) -> tuple[float, float]:
        xs, ys = warp_transform("EPSG:4326", self.crs, [lon], [lat])
        col, row = ~self.transform * (xs[0], ys[0])
        return col, row

    def cell_centres_lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        cols, rows = np.meshgrid(np.arange(self.width) + 0.5, np.arange(self.height) + 0.5)
        xs, ys = self.transform * (cols.ravel(), rows.ravel())
        lons, lats = warp_transform(self.crs, "EPSG:4326", list(xs), list(ys))
        return np.asarray(lons).reshape(self.shape), np.asarray(lats).reshape(self.shape)

    def to_json(self) -> dict:
        return {
            "crs": self.crs,
            "transform": list(self.transform)[:6],
            "width": self.width,
            "height": self.height,
            "dx": self.dx,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Grid":
        return cls(d["crs"], Affine(*d["transform"]), d["width"], d["height"], d["dx"])

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Grid":
        return cls.from_json(json.loads(path.read_text()))
