// OSM place lookup for the live simulation: names what-if results and lets
// the assistant point the map at a place. places.png is 16-bit grey with the
// pipeline's place id + 1 per cell (0 = none); places.json holds the names.

import { decode } from "fast-png";
import { NO_ETA, type JamReadback } from "../../sim/engine.ts";

export interface PlaceName {
  name: string;
  name_ml: string | null;
}

export interface PlaceIndex {
  ids: Uint16Array;
  names: PlaceName[];
  width: number;
}

/** Same thresholds as pipeline/hyacinth/aggregate.py. */
const EXISTING_HOURS = 1;
const MIN_CELLS = 4;

export async function loadPlaces(base: string, width: number, height: number): Promise<PlaceIndex | null> {
  try {
    const [png, names] = await Promise.all([
      fetch(`${base}/places.png`).then((r) => (r.ok && !(r.headers.get("content-type") ?? "").includes("html") ? r.arrayBuffer() : null)),
      fetch(`${base}/places.json`).then((r) => (r.ok && (r.headers.get("content-type") ?? "").includes("json") ? r.json() : null)),
    ]);
    if (!png || !names) return null;
    const img = decode(new Uint8Array(png));
    if (img.width !== width || img.height !== height || img.channels !== 1) return null;
    const ids = img.depth === 16 ? (img.data as Uint16Array) : Uint16Array.from(img.data as Uint8Array);
    return { ids, names: names as PlaceName[], width };
  } catch (e) {
    console.warn("places:", e);
    return null;
  }
}

/** Find a place by exact (case-insensitive) English or Malayalam name; returns its grid centroid. */
export function locatePlace(index: PlaceIndex, query: string, cellType: ArrayLike<number>): { place: PlaceName; x: number; y: number } | null {
  const q = query.trim().toLowerCase();
  const id = index.names.findIndex((p) => p.name.toLowerCase() === q || (p.name_ml ?? "").toLowerCase() === q);
  if (id < 0) return null;
  // Centroid of the place's water cells (fall back to all its cells).
  let sx = 0, sy = 0, k = 0, ax = 0, ay = 0, ak = 0;
  const w = index.width;
  for (let c = 0; c < index.ids.length; c++) {
    if (index.ids[c] !== id + 1) continue;
    const x = c % w;
    const y = (c - x) / w;
    ax += x; ay += y; ak++;
    if (cellType[c] !== 0) {
      sx += x; sy += y; k++;
    }
  }
  if (k) return { place: index.names[id], x: sx / k + 0.5, y: sy / k + 0.5 };
  if (ak) return { place: index.names[id], x: ax / ak + 0.5, y: ay / ak + 0.5 };
  return null;
}

export interface PlaceJam {
  place: string;
  place_ml: string | null;
  new_jam_ha: number;
  already_jammed_ha: number;
  /** Hours after the forecast start by which half of the new jam is in place. */
  new_jam_eta_hours: number | null;
}

export interface JamSummary {
  total_new_jam_ha: number;
  total_already_jammed_ha: number;
  places: PlaceJam[];
}

/** Jammed area and timing per named place, from a live run's jam/ETA readback. */
export function summarizeJams(jam: JamReadback, index: PlaceIndex | null, stepsPerHour: number, dxM: number, top = 12): JamSummary {
  const ha = (dxM * dxM) / 1e4;
  const per = new Map<number, { fresh: number[]; old: number }>();
  let fresh = 0;
  let old = 0;
  for (let c = 0; c < jam.eta.length; c++) {
    const e = jam.eta[c];
    if (e === NO_ETA) continue;
    const hours = e / stepsPerHour;
    const isNew = hours >= EXISTING_HOURS;
    if (isNew) fresh++;
    else old++;
    const id = index ? index.ids[c] - 1 : -1;
    if (id < 0) continue;
    let p = per.get(id);
    if (!p) per.set(id, (p = { fresh: [], old: 0 }));
    if (isNew) p.fresh.push(hours);
    else p.old++;
  }
  const places: PlaceJam[] = [];
  for (const [id, p] of per) {
    if (p.fresh.length + p.old < MIN_CELLS || !index) continue;
    p.fresh.sort((a, b) => a - b);
    const name = index.names[id];
    places.push({
      place: name.name,
      place_ml: name.name_ml,
      new_jam_ha: round1(p.fresh.length * ha),
      already_jammed_ha: round1(p.old * ha),
      new_jam_eta_hours: p.fresh.length ? Math.round(p.fresh[Math.floor(p.fresh.length / 2)]) : null,
    });
  }
  places.sort((a, b) => b.new_jam_ha - a.new_jam_ha || b.already_jammed_ha - a.already_jammed_ha);
  return { total_new_jam_ha: round1(fresh * ha), total_already_jammed_ha: round1(old * ha), places: places.slice(0, top) };
}

const round1 = (v: number) => Math.round(v * 10) / 10;
