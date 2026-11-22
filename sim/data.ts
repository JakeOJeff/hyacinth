// The contract between the Python pipeline and the simulation:
//
//   mask.png  RGB, one pixel per cell, row 0 = north. Exact colours:
//               (0,0,0)       land
//               (255,255,255) water (lake, canals, rivers)
//               (0,255,0)     river inflow   (pressure inlet)
//               (0,0,255)     sea outlet     (pressure outlet, mats flushed)
//               (255,0,0)     Thanneermukkom barrage (wall when closed)
//             A single-channel mask is read as water > 127.
//   seed.png  8-bit grey, same size: hyacinth mat cover 0..255 per cell.
//   wind.json WindForcing below.
//
// observed.png (optional, hindcast only) has the same format as seed.png and
// is drawn as an outline for comparison.

import { decode } from "fast-png";

export interface WindForcing {
  /** ISO time of t0 (normally the satellite acquisition, floored to the hour). */
  start: string;
  /** ISO time the forecast window begins; hours before it are spin-up. */
  forecast_start: string;
  /** Length of the forecast window in hours. */
  forecast_hours: number;
  step_hours: number;
  /** Wind toward east / north, m/s, one value per step from t0. */
  u: number[];
  v: number[];
  /** Grid spacing in metres. */
  dx_m: number;
  barrage_open: boolean;
  source: string;
  scene_date?: string | null;
  synthetic?: boolean;
}

export interface SimData {
  width: number;
  height: number;
  cellType: Uint32Array<ArrayBuffer>;
  seed: Uint8Array<ArrayBuffer>;
  observed?: Uint8Array<ArrayBuffer>;
  wind: WindForcing;
}

export const CELL = { LAND: 0, WATER: 1, INFLOW: 2, OUTLET: 3, BARRAGE: 4 } as const;

const PALETTE: Array<[number, number, number, number]> = [
  [0, 0, 0, CELL.LAND],
  [255, 255, 255, CELL.WATER],
  [0, 255, 0, CELL.INFLOW],
  [0, 0, 255, CELL.OUTLET],
  [255, 0, 0, CELL.BARRAGE],
];

interface Raster {
  width: number;
  height: number;
  channels: number;
  /** 8-bit samples, channels interleaved. */
  data: Uint8Array;
}

export function decodeRaster(bytes: Uint8Array): Raster {
  const png = decode(bytes);
  if (png.palette) throw new Error("palette PNGs are not supported; write RGB or grey");
  let data: Uint8Array;
  if (png.depth === 8) {
    data = png.data as Uint8Array;
  } else if (png.depth === 16) {
    const src = png.data as Uint16Array;
    data = new Uint8Array(src.length);
    for (let i = 0; i < src.length; i++) data[i] = src[i] >> 8;
  } else {
    throw new Error(`unsupported PNG bit depth ${png.depth}`);
  }
  return { width: png.width, height: png.height, channels: png.channels, data };
}

export function maskToCellTypes(r: Raster): Uint32Array<ArrayBuffer> {
  const n = r.width * r.height;
  const out = new Uint32Array(n);
  const ch = r.channels;
  for (let i = 0; i < n; i++) {
    if (ch < 3) {
      out[i] = r.data[i * ch] > 127 ? CELL.WATER : CELL.LAND;
      continue;
    }
    const R = r.data[i * ch];
    const G = r.data[i * ch + 1];
    const B = r.data[i * ch + 2];
    let best = 0;
    let bestD = Infinity;
    for (const [pr, pg, pb, t] of PALETTE) {
      const d = (R - pr) ** 2 + (G - pg) ** 2 + (B - pb) ** 2;
      if (d < bestD) {
        bestD = d;
        best = t;
      }
    }
    out[i] = best;
  }
  return out;
}

export function rasterToGrey(r: Raster): Uint8Array<ArrayBuffer> {
  const n = r.width * r.height;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = r.data[i * r.channels];
  return out;
}

export function validateWind(w: WindForcing): WindForcing {
  if (!Array.isArray(w.u) || !Array.isArray(w.v) || w.u.length !== w.v.length || w.u.length === 0) {
    throw new Error("wind.json: u and v must be equal-length non-empty arrays");
  }
  if (!(w.dx_m > 0)) throw new Error("wind.json: dx_m must be > 0");
  if (!(w.step_hours > 0)) throw new Error("wind.json: step_hours must be > 0");
  if (Number.isNaN(Date.parse(w.start)) || Number.isNaN(Date.parse(w.forecast_start))) {
    throw new Error("wind.json: start / forecast_start must be ISO timestamps");
  }
  return w;
}

/** Hours from t0 to the start of the forecast window. */
export function spinupHours(w: WindForcing): number {
  return (Date.parse(w.forecast_start) - Date.parse(w.start)) / 3_600_000;
}

export function buildSimData(
  maskPng: Uint8Array,
  seedPng: Uint8Array,
  wind: WindForcing,
  observedPng?: Uint8Array,
): SimData {
  const mask = decodeRaster(maskPng);
  const seed = decodeRaster(seedPng);
  if (mask.width !== seed.width || mask.height !== seed.height) {
    throw new Error(`mask ${mask.width}x${mask.height} and seed ${seed.width}x${seed.height} differ in size`);
  }
  let observed: Uint8Array<ArrayBuffer> | undefined;
  if (observedPng) {
    const o = decodeRaster(observedPng);
    if (o.width !== mask.width || o.height !== mask.height) throw new Error("observed.png size differs from mask");
    observed = rasterToGrey(o);
  }
  return {
    width: mask.width,
    height: mask.height,
    cellType: maskToCellTypes(mask),
    seed: rasterToGrey(seed),
    observed,
    wind: validateWind(wind),
  };
}
