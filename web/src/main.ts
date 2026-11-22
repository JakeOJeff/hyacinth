import { buildSimData, spinupHours, type WindForcing } from "../../sim/data.ts";
import { HyacinthSim, type ShaderSources } from "../../sim/engine.ts";
import common from "../../sim/shaders/common.wgsl?raw";
import lbm from "../../sim/shaders/lbm.wgsl?raw";
import particles from "../../sim/shaders/particles.wgsl?raw";
import accumulate from "../../sim/shaders/accumulate.wgsl?raw";
import render from "../../sim/shaders/render.wgsl?raw";
import lake3dWgsl from "./lake3d.wgsl?raw";
import { Lake3D } from "./lake3d.ts";
import { FramePacer } from "./pacer.ts";
import { loadPlaces, locatePlace, summarizeJams, type JamSummary } from "./places.ts";
import { describeScenario, initAsk, type AskTools } from "./ask.ts";
import { initChrome, moveTabThumb, revealMap } from "./ui.ts";
import { healthTools, loadHealth, renderHealth, type HealthZone } from "./health.ts";

const shaders: ShaderSources = { common, lbm, particles, accumulate, render };
const IST_OFFSET_MS = 5.5 * 3600 * 1000;
const SVG_NS = "http://www.w3.org/2000/svg";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`missing #${id}`);
  return el as T;
};

interface Caption {
  ml: string;
  en: string;
  text: string;
  source: string;
}

interface IndexEntry {
  name: string;
  source?: string;
  scene_date?: string | null;
  observed_date?: string | null;
  synthetic?: boolean;
}

interface Segment {
  place: string;
  place_ml: string | null;
  waterway: string | null;
  new_jam_area_ha: number;
  already_jammed_area_ha: number;
  eta_hours: number;
  by_time_ist: string;
}

interface Facts {
  synthetic: boolean;
  forecast_start_ist: string;
  window_hours: number;
  barrage: "open" | "closed";
  wind: { mean_speed_ms: number; max_speed_ms: number; prevailing_from_deg: number; prevailing_from: string };
  total_new_jam_area_ha: number;
  segments: Segment[];
}

type Mode = "3d" | "live" | "timelapse" | "heatmap" | "hotspots" | "latest";

function notice(msg: string) {
  const n = $("notice");
  n.textContent = msg;
  n.hidden = false;
}

async function fetchBytes(url: string): Promise<Uint8Array> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

async function fetchJson<T>(url: string): Promise<T | null> {
  const r = await fetch(url);
  if (!r.ok) return null;
  const type = r.headers.get("content-type") ?? "";
  if (!type.includes("json")) return null; // dev server returns index.html for missing files
  return (await r.json()) as T;
}

async function exists(url: string): Promise<boolean> {
  const r = await fetch(url, { method: "HEAD" });
  const type = r.headers.get("content-type") ?? "";
  return r.ok && !type.includes("text/html");
}

const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

function istParts(ms: number) {
  const d = new Date(ms + IST_OFFSET_MS);
  return {
    day: DAYS[d.getUTCDay()],
    date: d.getUTCDate(),
    month: d.toLocaleString("en", { month: "short", timeZone: "UTC" }),
    hm: `${String(d.getUTCHours()).padStart(2, "0")}:${String(d.getUTCMinutes()).padStart(2, "0")}`,
  };
}

function compass(deg: number): string {
  const names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
  return names[Math.round(((deg % 360) + 360) % 360 / 45) % 8];
}

async function pickDataset(): Promise<{ name: string; entries: IndexEntry[] }> {
  const want = new URLSearchParams(location.search).get("data");
  const entries = (await fetchJson<IndexEntry[]>("./data/index.json")) ?? [];
  if (want) return { name: want, entries };
  for (const pref of ["latest", "hindcast", "synthetic"]) {
    if (entries.some((e) => e.name === pref)) return { name: pref, entries };
  }
  return { name: entries[0]?.name ?? "synthetic", entries };
}

function renderDatasetSwitcher(current: string, entries: IndexEntry[]) {
  const box = $("dataset");
  box.replaceChildren();
  if (entries.length < 2) return;
  for (const e of entries) {
    const a = document.createElement("a");
    a.href = `?data=${encodeURIComponent(e.name)}`;
    a.textContent = e.synthetic ? `${e.name} (test)` : e.name;
    if (e.name === current) a.setAttribute("aria-current", "page");
    box.append(a);
  }
}

/** Keep the filled part of a range track in sync with its value. */
function paintRange(el: HTMLInputElement) {
  const min = Number(el.min || 0);
  const max = Number(el.max || 100);
  el.style.setProperty("--p", `${((Number(el.value) - min) / (max - min)) * 100}%`);
}

// ---------- static forecast panels ----------

function renderHeader(facts: Facts | null, wind: WindForcing | null) {
  const issued = facts?.forecast_start_ist.match(/^(\w+ \d+ \w+) \d{4} (\d\d:\d\d)/);
  if (issued) $("issued").textContent = `${issued[1]}, ${issued[2]}`;
  else if (wind) {
    const p = istParts(Date.parse(wind.forecast_start));
    $("issued").textContent = `${p.day} ${p.date} ${p.month}, ${p.hm}`;
  }
  const hours = facts?.window_hours ?? wind?.forecast_hours;
  if (hours) $("horizon").textContent = `${hours} h`;
  const open = facts ? facts.barrage === "open" : wind?.barrage_open;
  if (open !== undefined) {
    const b = $("barrage-state");
    b.textContent = open ? "Open" : "Closed";
    b.className = open ? "blue" : "orange";
  }
  $("synthetic-badge").hidden = !(facts?.synthetic ?? (wind as { synthetic?: boolean } | null)?.synthetic);
}

function renderWind(facts: Facts | null, wind: WindForcing | null) {
  const speeds: number[] = [];
  if (wind) {
    const offset = Math.max(0, Math.round((Date.parse(wind.forecast_start) - Date.parse(wind.start)) / 3600_000 / wind.step_hours));
    const n = Math.round(wind.forecast_hours / wind.step_hours);
    for (let i = offset; i <= offset + n && i < wind.u.length; i++) speeds.push(Math.hypot(wind.u[i], wind.v[i]));
  }
  const mean = facts?.wind.mean_speed_ms ?? (speeds.length ? speeds.reduce((a, b) => a + b, 0) / speeds.length : NaN);
  if (Number.isFinite(mean)) $("wind-avg").textContent = mean.toFixed(1);
  if (facts) {
    $("wind-dir").textContent = compass(facts.wind.prevailing_from_deg);
    $("wind-arrow").style.transform = `rotate(${(facts.wind.prevailing_from_deg + 180) % 360}deg)`;
  }

  const svg = $("spark") as unknown as SVGSVGElement;
  svg.replaceChildren();
  if (speeds.length < 2) return;
  const W = 300;
  const H = 70;
  const top = Math.max(...speeds, 1) * 1.15;
  const pts = speeds.map((s, i) => [(i / (speeds.length - 1)) * W, H - 6 - (s / top) * (H - 12)] as const);
  const d = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join("");
  const area = document.createElementNS(SVG_NS, "path");
  area.setAttribute("d", `${d}L${W},${H}L0,${H}Z`);
  area.style.fill = "color-mix(in srgb, var(--chart) 14%, transparent)";
  const line = document.createElementNS(SVG_NS, "path");
  line.setAttribute("d", d);
  line.setAttribute("fill", "none");
  line.style.stroke = "var(--chart)";
  line.setAttribute("stroke-width", "2");
  line.setAttribute("vector-effect", "non-scaling-stroke");
  line.setAttribute("stroke-linejoin", "round");
  const meanY = H - 6 - (mean / top) * (H - 12);
  const avg = document.createElementNS(SVG_NS, "line");
  avg.setAttribute("x1", "0");
  avg.setAttribute("x2", String(W));
  avg.setAttribute("y1", meanY.toFixed(1));
  avg.setAttribute("y2", meanY.toFixed(1));
  avg.style.stroke = "var(--muted)";
  avg.setAttribute("stroke-width", "1.5");
  avg.setAttribute("stroke-dasharray", "4 4");
  avg.setAttribute("vector-effect", "non-scaling-stroke");
  const title = document.createElementNS(SVG_NS, "title");
  title.textContent = `Wind speed over the forecast window, peak ${Math.max(...speeds).toFixed(1)} m/s; dashed = average`;
  svg.append(title, area, avg, line);
}

function renderRisk(facts: Facts | null) {
  if (!facts) return;
  $("risk-places").textContent = String(facts.segments.length);
  $("risk-area").textContent = facts.total_new_jam_area_ha.toFixed(1);
  const fresh = facts.total_new_jam_area_ha;
  const old = facts.segments.reduce((a, s) => a + s.already_jammed_area_ha, 0);
  const SLOTS = 36;
  const unit = Math.max(1, Math.ceil((fresh + old) / SLOTS));
  const nFresh = Math.round(fresh / unit);
  const nOld = Math.min(SLOTS - nFresh, Math.round(old / unit));
  const box = $("dots");
  box.replaceChildren();
  for (let i = 0; i < SLOTS; i++) {
    const dot = document.createElement("i");
    dot.style.setProperty("--i", String(i));
    if (i < nFresh) dot.className = "orange";
    else if (i < nFresh + nOld) dot.className = "lime";
    box.append(dot);
  }
  box.parentElement!.querySelector(".dots-key span")!.lastChild!.textContent = `new (1 dot = ${unit} ha)`;
}

function renderTimeline(facts: Facts | null) {
  if (!facts || facts.segments.length === 0) return;
  const horizon = facts.window_hours || 72;
  const box = $("timeline");
  box.replaceChildren();
  const segs = [...facts.segments].sort((a, b) => a.eta_hours - b.eta_hours);
  const first = segs[0];
  const fw = first.by_time_ist.match(/^(\w+) \d+ \w+ (\d\d:\d\d)/);
  $("next-block").textContent = `${first.place.replace(/ \(synthetic\)$/, "")}, ${fw ? `${fw[1]} ${fw[2]}` : `+${first.eta_hours} h`}`;
  for (const s of segs) {
    const row = document.createElement("div");
    row.className = "tl-row";
    const place = document.createElement("div");
    place.className = "place";
    place.textContent = s.place.replace(/ \(synthetic\)$/, "");
    place.title = s.place;
    if (s.place_ml) {
      const ml = document.createElement("small");
      ml.className = "ml";
      ml.textContent = s.place_ml;
      place.append(ml);
    }
    const track = document.createElement("div");
    track.className = "track";
    const bar = document.createElement("div");
    const color = s.eta_hours <= 24 ? "orange" : s.eta_hours <= 48 ? "lime" : "blue";
    bar.className = `tl-bar ${color}`;
    bar.style.setProperty("--i", String(box.childElementCount));
    bar.style.left = `${Math.min(100, Math.max(0, (s.eta_hours / horizon) * 100))}%`;
    const when = s.by_time_ist.match(/^(\w+) \d+ \w+ (\d\d:\d\d)/);
    const label = document.createElement("span");
    label.textContent = when ? `${when[1]} ${when[2]}` : `+${s.eta_hours} h`;
    const ha = document.createElement("b");
    ha.textContent = s.new_jam_area_ha.toFixed(1);
    bar.title = `${s.place}: likely blocked by ${s.by_time_ist} (+${s.eta_hours} h), ${s.new_jam_area_ha} ha new jam, ${s.already_jammed_area_ha} ha already jammed`;
    bar.append(label, ha);
    track.append(bar);
    row.append(place, track);
    box.append(row);
  }
}

async function renderCaption(base: string) {
  const cap = await fetchJson<Caption>(`${base}/caption.json`);
  if (!cap) return;
  $("caption-card").hidden = false;
  $("cap-ml").textContent = cap.ml;
  $("cap-en").textContent = cap.en;
  const meta = cap.text.split("\n").filter(Boolean);
  $("cap-meta").textContent = meta[meta.length - 1] ?? "";
}

// ---------- main ----------

interface Stats {
  state_step?: number;
  seeded_particles?: number;
}

/** Slider 0..100 -> simulated hours per wall-clock second (log scale, 0.25 .. 24). */
const rateFromSlider = (v: number) => 0.25 * Math.pow(96, v / 100);

async function fetchOptionalBytes(url: string): Promise<Uint8Array | null> {
  const r = await fetch(url).catch(() => null);
  if (!r || !r.ok || (r.headers.get("content-type") ?? "").includes("text/html")) return null;
  return new Uint8Array(await r.arrayBuffer());
}

async function main() {
  initChrome($("stage"));
  const { name, entries } = await pickDataset();
  renderDatasetSwitcher(name, entries);
  const base = `./data/${encodeURIComponent(name)}`;

  const [facts, wind, stats, hasVideo, hasHeat, health] = await Promise.all([
    fetchJson<Facts>(`${base}/facts.json`).catch(() => null),
    fetchJson<WindForcing>(`${base}/wind.json`).catch(() => null),
    fetchJson<Stats>(`${base}/stats.json`).catch(() => null),
    exists(`${base}/forecast.mp4`).catch(() => false),
    exists(`${base}/heatmap.png`).catch(() => false),
    loadHealth(),
  ]);
  renderHeader(facts, wind);
  renderWind(facts, wind);
  renderRisk(facts);
  renderTimeline(facts);
  $("t-next").textContent = $("next-block").textContent || "None forecast";
  renderCaption(base).catch((e) => console.warn("caption:", e));
  if (wind && !wind.barrage_open) {
    const b = $("ask-barrage");
    b.dataset.ask = "What happens to the jams if the Thanneermukkom barrage is opened for the next 48 hours?";
  }

  // View modes: live canvas, the daily timelapse video, the daily heatmap image.
  const stage = $("stage");
  const canvas = $<HTMLCanvasElement>("canvas");
  const video = $<HTMLVideoElement>("video");
  const heatImg = $<HTMLImageElement>("heatmap");
  const hotImg = $<HTMLImageElement>("hot-img");
  const lastImg = $<HTMLImageElement>("last-img");
  if (health) {
    hotImg.src = `./data/health/${health.images.hotspots}`;
    if (health.images.latest) lastImg.src = `./data/health/${health.images.latest}`;
  }
  if (hasVideo) video.src = `${base}/forecast.mp4`;
  if (hasHeat) heatImg.src = `${base}/heatmap.png`;
  const modeButtons = [...document.querySelectorAll<HTMLButtonElement>("[data-mode]")].filter((b) => b.tagName === "BUTTON");
  const canvas3d = $<HTMLCanvasElement>("canvas3d");
  let sim: HyacinthSim | null = null;
  let lake: Lake3D | null = null;
  let mode: Mode = "live";
  const HINTS: Record<Mode, string> = {
    "3d": "Drag to turn · pinch or scroll to zoom · tap water to drop weed",
    live: "Tap the water to drop weed",
    timelapse: "Today's 72-hour forecast as a video",
    heatmap: "Where weed stays stuck longest",
    hotspots: "How often weed and possible rubbish were seen since Jan 2025",
    latest: health?.latest_pass ? `Satellite photo from ${health.latest_pass}` : "Last clear satellite photo",
  };

  const setMode = (m: Mode) => {
    if (m === "3d" && !lake) m = "live";
    mode = m;
    stage.dataset.mode = m;
    for (const b of modeButtons) b.setAttribute("aria-selected", String(b.dataset.mode === m));
    moveTabThumb();
    const liveHeat = m === "heatmap" && !hasHeat; // fall back to the sim's own heatmap
    canvas3d.hidden = m !== "3d";
    canvas.hidden = !(m === "live" || liveHeat);
    video.hidden = m !== "timelapse";
    heatImg.hidden = !(m === "heatmap" && hasHeat);
    hotImg.hidden = m !== "hotspots";
    lastImg.hidden = m !== "latest";
    if (m === "timelapse") video.play().catch(() => {});
    else video.pause();
    if (sim) sim.viewMode = liveHeat ? 1 : 0;
    $("legend").hidden = m !== "live";
    $("legend-health").hidden = m !== "hotspots" && m !== "latest";
    $("ctl3d").hidden = m !== "3d";
    $("hint").textContent = HINTS[m];
  };
  for (const b of modeButtons) {
    const m = b.dataset.mode as Mode;
    if ((m === "timelapse" && !hasVideo) || m === "3d") {
      b.disabled = true;
      b.title = m === "3d" ? "Needs WebGPU" : "No timelapse for this dataset";
    }
    if ((m === "hotspots" && !health) || (m === "latest" && !health?.images.latest)) {
      b.disabled = true;
      b.title = "Lake-health dataset not published";
    }
    b.addEventListener("click", () => setMode(m));
  }
  setMode("live");
  for (const r of document.querySelectorAll<HTMLInputElement>("input[type=range]")) paintRange(r);

  const clockPhase = $("clock-phase");
  const clockTime = $("clock-time");
  const clockMeta = $("clock-meta");
  const clockBar = $("clock-bar");

  // ---- map pin (assistant / dashboard), works on every view ----
  const pin = $<HTMLButtonElement>("pin");
  let pinAt: { x: number; y: number } | null = null;
  pin.addEventListener("click", () => {
    pinAt = null;
    pin.hidden = true;
  });
  /** Screen rect of an object-fit: contain image's picture. */
  const contained = (img: HTMLImageElement) => {
    const r = img.getBoundingClientRect();
    const iw = img.naturalWidth || 1;
    const ih = img.naturalHeight || 1;
    const k = Math.min(r.width / iw, r.height / ih);
    const w = iw * k;
    const hh = ih * k;
    return { left: r.left + (r.width - w) / 2, top: r.top + (r.height - hh) / 2, width: w, height: hh };
  };
  const gridSize = () => (sim ? [sim.width, sim.height] : [canvas.width || 746, canvas.height || 1054]) as [number, number];
  const placePin = () => {
    requestAnimationFrame(placePin);
    if (!pinAt) return;
    const [gw, gh] = gridSize();
    let xy: [number, number] | null = null;
    if (mode === "3d" && lake) xy = lake.project(pinAt.x, pinAt.y);
    else {
      const target = mode === "live" ? canvas : mode === "hotspots" ? hotImg : mode === "latest" ? lastImg : mode === "heatmap" && !heatImg.hidden ? heatImg : null;
      if (target && !target.hidden) {
        const r = target instanceof HTMLImageElement ? contained(target) : target.getBoundingClientRect();
        xy = [r.left + (pinAt.x / gw) * r.width, r.top + (pinAt.y / gh) * r.height];
      }
    }
    const hr = stage.getBoundingClientRect();
    const inside = xy && xy[0] >= hr.left && xy[0] <= hr.right && xy[1] >= hr.top && xy[1] <= hr.bottom;
    pin.hidden = !inside;
    if (inside) pin.style.transform = `translate(${xy![0].toFixed(1)}px, ${xy![1].toFixed(1)}px)`;
  };
  requestAnimationFrame(placePin);
  const pinTo = (x: number, y: number, label: string) => {
    pinAt = { x, y };
    $("pin-label").textContent = label;
    lake?.flyTo(x, y, 160);
    revealMap();
  };
  const showLayer = (l: "hotspots" | "latest" | "forecast") => {
    setMode(l === "forecast" ? (lake ? "3d" : "live") : l);
    revealMap();
  };
  const showZone = (z: HealthZone) => {
    if (z.x === null || z.y === null) return;
    if (mode === "timelapse") setMode(lake ? "3d" : "live");
    pinTo(z.x, z.y, z.name_ml ? `${z.name} · ${z.name_ml}` : z.name);
  };
  const dataTools = health ? healthTools(health, { showLayer }) : {};
  if (health) {
    $("health-missing").hidden = true;
    renderHealth(health, { showZone, showLayer });
  }

  const fallBackToMedia = (msg: string) => {
    notice(msg);
    $("clock").hidden = true;
    if (hasVideo) setMode("timelapse");
    else if (health) setMode("hotspots");
    initAsk(name, { extra: dataTools });
  };

  if (!("gpu" in navigator)) {
    fallBackToMedia("This browser has no WebGPU. Try a recent Chrome or Edge; the daily timelapse still works.");
    return;
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    fallBackToMedia("WebGPU is available but no GPU adapter was found.");
    return;
  }
  // Only ask for what the sim needs: requesting the adapter's maximum buffer
  // size makes some drivers reserve far more memory than the ~100 MB used.
  const NEED = 512 * 1024 * 1024;
  const newDevice = async (a: GPUAdapter) => {
    const d = await a.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(NEED, a.limits.maxStorageBufferBindingSize),
        maxBufferSize: Math.min(NEED, a.limits.maxBufferSize),
      },
    });
    d.lost.then((info) => {
      if (info.reason !== "destroyed") notice(`The GPU stopped responding (${info.message || info.reason}). This usually means the computer is low on memory: close some tabs or apps and reload.`);
    });
    return d;
  };
  let device = await newDevice(adapter);

  let hasObserved = false;
  let snapshot: Float32Array | null = null;
  try {
    if (!wind) throw new Error(`no dataset "${name}" (wind.json missing)`);
    const [mask, seed, observed, state] = await Promise.all([
      fetchBytes(`${base}/mask.png`),
      fetchBytes(`${base}/seed.png`),
      fetchOptionalBytes(`${base}/observed.png`),
      fetchOptionalBytes(`${base}/state.bin`),
    ]);
    const data = buildSimData(mask, seed, wind, observed ?? undefined);
    try {
      sim = await HyacinthSim.create(device, data, shaders);
    } catch (e) {
      // "Instance dropped" / device lost: the GPU process restarted. Retry once on a fresh device.
      if (!/dropped|lost/i.test((e as Error).message)) throw e;
      const a2 = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!a2) throw new Error("the GPU stopped responding (often low memory); close some tabs or apps and reload");
      device = await newDevice(a2);
      sim = await HyacinthSim.create(device, data, shaders);
    }
    if (state && state.byteLength % 16 === 0) snapshot = new Float32Array(state.buffer, state.byteOffset, state.byteLength / 4);
    hasObserved = !!observed;
    sim.showObserved = hasObserved;
    if (hasObserved) {
      $("obs-wrap").hidden = false;
      $("legend-obs").hidden = false;
    }
  } catch (e) {
    fallBackToMedia(`Could not start the simulation: ${(e as Error).message}`);
    throw e;
  }
  const s = sim;
  setMode(mode);
  const places = await loadPlaces(base, s.width, s.height);

  canvas.width = s.width;
  canvas.height = s.height;
  canvas.style.aspectRatio = `${s.width} / ${s.height}`;
  stage.style.setProperty("--ar", String(s.width / s.height));
  const ctx = canvas.getContext("webgpu");
  if (!ctx) {
    notice("Could not get a WebGPU canvas context.");
    return;
  }
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "opaque" });

  const t0 = Date.parse(wind.start);
  const spinup = spinupHours(wind);
  const forecastStep = Math.round(spinup * s.stepsPerHour);
  const endStep = forecastStep + Math.round(wind.forecast_hours * s.stepsPerHour);
  let playing = true;

  // Controls
  const speed = $<HTMLInputElement>("speed");
  const manual = $<HTMLInputElement>("wind-manual");
  const dir = $<HTMLInputElement>("dir");
  const ws = $<HTMLInputElement>("ws");
  const barrage = $<HTMLInputElement>("barrage");
  const river = $<HTMLInputElement>("river");
  const drop = $<HTMLInputElement>("drop");
  const observed = $<HTMLInputElement>("observed");
  const playBtn = $<HTMLButtonElement>("play");
  const ranges = [speed, dir, ws, river, drop];
  const controls = $("sim-controls");

  barrage.checked = s.barrageOpen;
  river.value = String(Math.round(s.physParams.riverHead * 1000));
  const todayBarrage = s.barrageOpen;

  let rate = rateFromSlider(Number(speed.value));
  let gpuLimited = false;
  const sync = () => {
    rate = rateFromSlider(Number(speed.value));
    $("speed-out").textContent = `${rate < 1 ? rate.toFixed(2) : rate.toFixed(1)} h/s${gpuLimited ? " · GPU max" : ""}`;
    dir.disabled = ws.disabled = !manual.checked;
    s.windOverride = manual.checked ? { speed: Number(ws.value), dirFrom: Number(dir.value) } : null;
    $("dir-out").textContent = `${dir.value}° ${compass(Number(dir.value))}`;
    $("ws-out").textContent = `${Number(ws.value).toFixed(1)} m/s`;
    s.barrageOpen = barrage.checked;
    s.tune({ riverHead: Number(river.value) / 1000 });
    $("river-out").textContent = Number(river.value) === 0 ? "off" : `head ${river.value}`;
    $("drop-out").textContent = `${drop.value} mats`;
    s.showObserved = observed.checked && hasObserved;
    for (const r of ranges) paintRange(r);
  };
  for (const el of [...ranges, manual, barrage, observed]) el.addEventListener("input", sync);
  sync();

  const setPlaying = (p: boolean) => {
    playing = p;
    playBtn.textContent = playing ? "Pause" : "Play";
  };
  playBtn.addEventListener("click", () => setPlaying(!playing));

  const dropAt = (x: number, y: number) => {
    if (job) return;
    const radius = Math.max(2, Math.sqrt(Number(drop.value) / s.particlesPerSeedCell / Math.PI));
    s.dropHyacinth(x, y, radius, Number(drop.value));
  };
  canvas.addEventListener("pointerdown", (ev) => {
    if (mode !== "live") return;
    const r = canvas.getBoundingClientRect();
    dropAt(((ev.clientX - r.left) / r.width) * s.width, ((ev.clientY - r.top) / r.height) * s.height);
  });

  // ---- fast-forward jobs: warm-up, catch-up, what-if runs ----
  // A job runs the sim as fast as the GPU allows while still drawing frames.
  interface Job {
    target: number;
    from: number;
    particles: boolean;
    label: string;
    progress?: (fraction: number, label: string) => void;
    resolve: () => void;
  }
  let job: Job | null = null;
  const fastForward = (target: number, particles: boolean, label: string, progress?: Job["progress"]) =>
    new Promise<void>((resolve) => {
      job = { target, from: s.step, particles, label, progress, resolve };
    });

  // Start at the forecast start. With the daily render's snapshot of the mats
  // at that moment, only the currents need a short warm-up (mats frozen);
  // otherwise replay the whole spin-up from the satellite pass.
  const start = async () => {
    const warm = Math.min(forecastStep, Math.round(2 * s.stepsPerHour));
    const usable = snapshot && stats?.state_step === forecastStep && stats.seeded_particles === s.seedParticles;
    if (usable && s.loadParticles(snapshot!, forecastStep - warm)) {
      await fastForward(forecastStep, false, "Warming up currents");
    } else if (forecastStep > 0) {
      await fastForward(forecastStep, true, `Catching up from the satellite pass (${spinup.toFixed(0)} h)`);
    }
    s.clearJam();
    s.checkpoint();
  };
  const ready = start();

  const scenarioBox = $("scenario");
  const exitScenario = () => {
    scenarioBox.hidden = true;
    manual.checked = false;
    barrage.checked = todayBarrage;
    sync();
    s.restore();
    setPlaying(true);
  };
  $("scenario-exit").addEventListener("click", exitScenario);

  $("reset").addEventListener("click", async () => {
    if (job) return;
    scenarioBox.hidden = true;
    if (!s.restore()) {
      s.reset();
      await start();
    }
    setPlaying(true);
  });

  // ---- the assistant's tools ----
  const baselines = new Map<string, JamSummary>();
  const runCase = async (open: boolean, wind: { speed: number; dirFrom: number } | null, hours: number, label: string, progress: Job["progress"]) => {
    s.barrageOpen = open;
    s.windOverride = wind;
    s.restore();
    await fastForward(s.step + Math.round(hours * s.stepsPerHour), true, label, progress);
    return summarizeJams(await s.readJam(), places, s.stepsPerHour, s.data.wind.dx_m);
  };
  const tools: AskTools = {
    extra: dataTools,
    async runScenario(input, progress) {
      await ready;
      if (job) throw new Error("another run is in progress");
      const open = input.barrage === "as_forecast" ? todayBarrage : input.barrage === "open";
      const windFixed = input.wind === "fixed" ? { speed: input.wind_speed_ms, dirFrom: input.wind_from_deg } : null;
      if (mode !== "3d" && mode !== "live") setMode(lake ? "3d" : "live");
      revealMap();
      setPlaying(false);
      controls.inert = true;
      try {
        const key = `${input.hours}|${river.value}`;
        const both = open !== todayBarrage || windFixed !== null;
        let baseline = baselines.get(key);
        if (!baseline) {
          baseline = await runCase(todayBarrage, null, input.hours, both ? "Baseline (today's conditions)" : "Today's conditions", (f, l) =>
            progress(both ? f / 2 : f, l)
          );
          baselines.set(key, baseline);
        }
        const scenario = both
          ? await runCase(open, windFixed, input.hours, "What-if", (f, l) => progress(0.5 + f / 2, l))
          : baseline;
        // Leave the what-if on screen, with the controls showing its settings.
        barrage.checked = open;
        manual.checked = windFixed !== null;
        if (windFixed) {
          dir.value = String(Math.round(windFixed.dirFrom));
          ws.value = String(windFixed.speed);
        }
        sync();
        $("scenario-text").textContent = describeScenario(input);
        scenarioBox.hidden = false;
        return {
          forecast_start_ist: facts?.forecast_start_ist ?? null,
          hours: input.hours,
          baseline: { barrage: todayBarrage ? "open" : "closed", wind: "forecast", ...baseline },
          scenario: both
            ? {
              barrage: open ? "open" : "closed",
              wind: windFixed ? `fixed, from ${Math.round(windFixed.dirFrom)} deg at ${windFixed.speed} m/s` : "forecast",
              ...scenario,
            }
            : "same as baseline",
          note: places
            ? "Both runs were computed live with the same method. Numbers can differ a little from the daily forecast facts, which group jams by connected stretch."
            : "No place names are available for this dataset; only totals.",
        };
      } finally {
        controls.inert = false;
      }
    },
    showPlace(query) {
      if (!places) return { found: false, reason: "no place index for this dataset" };
      const hit = locatePlace(places, query, s.data.cellType);
      if (!hit) return { found: false, reason: `no place named "${query}"` };
      if (mode === "timelapse") setMode(lake ? "3d" : "live");
      pinTo(hit.x, hit.y, hit.place.name_ml ? `${hit.place.name} · ${hit.place.name_ml}` : hit.place.name);
      return { found: true, place: hit.place.name };
    },
  };
  initAsk(name, tools);

  // 3D view, drawn from the same GPU buffers. Optional: the 2D view still works without it.
  try {
    if (new URLSearchParams(location.search).has("no3d")) throw new Error("disabled by ?no3d");
    lake = await Lake3D.create(s, canvas3d, common, lake3dWgsl);
    lake.onPick = dropAt;
    for (const b of modeButtons) {
      if (b.dataset.mode === "3d") {
        b.disabled = false;
        b.title = "";
      }
    }
    const l = lake;
    const toggle = (id: string, get: () => boolean, set: (v: boolean) => void) => {
      const btn = $<HTMLButtonElement>(id);
      btn.setAttribute("aria-pressed", String(get()));
      btn.addEventListener("click", () => {
        set(!get());
        btn.setAttribute("aria-pressed", String(get()));
      });
    };
    toggle("glow", () => l.heatOn, (v) => (l.heatOn = v));
    toggle("orbit", () => l.autoOrbit, (v) => (l.autoOrbit = v));
    $("home").addEventListener("click", () => l.resetView());
    setMode("3d");
  } catch (e) {
    console.warn("3D view unavailable:", e);
  }

  // ---- frame loop ----
  // The pacer keeps at most two frames queued on the GPU and sizes the number
  // of sim steps per frame from measured GPU time, so drawing stays smooth on
  // any GPU: a slow machine just simulates fewer hours per second. While the
  // hero is off screen nothing is drawn, and normal playback pauses, but
  // warm-ups and what-if runs keep going.
  const pacer = new FramePacer(device);
  let heroVisible = true;
  new IntersectionObserver(([e]) => (heroVisible = e.isIntersecting)).observe(stage);
  let lastText = 0;
  let lastNow = performance.now();
  let carry = 0;

  const frame = (now: number) => {
    requestAnimationFrame(frame);
    const dt = Math.min(0.1, (now - lastNow) / 1000);
    lastNow = now;
    if (!pacer.beginFrame(now)) return;
    const j = job;
    if (!heroVisible && !j) return;

    let steps = 0;
    if (j) {
      steps = Math.min(pacer.stepBudget(), j.target - s.step);
      if (steps > 0) s.advance(steps, { particles: j.particles });
      const f = (s.step - j.from) / Math.max(1, j.target - j.from);
      j.progress?.(f, j.label);
      if (s.step >= j.target) {
        job = null;
        j.resolve();
      }
    } else if (playing) {
      const want = rate * s.stepsPerHour * dt + carry;
      let n = Math.floor(want);
      carry = want - n;
      if (!s.windOverride) n = Math.min(n, Math.max(0, endStep - s.step));
      const budget = pacer.stepBudget();
      gpuLimited = n > budget;
      if (gpuLimited) {
        n = budget;
        carry = 0;
      }
      if (n > 0) s.advance(n);
      steps = n;
    }

    if (heroVisible) {
      if (mode === "3d" && lake) {
        const w3 = s.currentWind();
        const istHour = (((t0 + s.simHours * 3600_000 + IST_OFFSET_MS) / 3600_000) % 24 + 24) % 24;
        lake.render({ hour: istHour, wind: { u: w3.u, v: w3.v, speed: w3.speed } });
      } else if (!canvas.hidden) s.render(ctx.getCurrentTexture().createView(), format);
    }
    // Fast-forwarding may use more of each frame: throughput over frame rate.
    pacer.endFrame(steps, j ? 0.9 : 0.7);

    if (now - lastText < 120) return;
    lastText = now;
    const w = s.currentWind();
    const t = istParts(t0 + s.simHours * 3600_000);
    const rel = s.simHours - spinup;
    const active = job;
    if (active) {
      const f = (s.step - active.from) / Math.max(1, active.target - active.from);
      clockPhase.textContent = active.label;
      clockBar.hidden = false;
      clockBar.style.setProperty("--p", `${(f * 100).toFixed(1)}%`);
    } else {
      clockBar.hidden = true;
      clockPhase.textContent = s.step >= endStep && !s.windOverride
        ? `Forecast +${rel.toFixed(0)} h · end (Reset to replay)`
        : `Forecast +${rel.toFixed(1)} h`;
    }
    clockPhase.classList.toggle("warn", !!active);
    clockTime.textContent = `${t.day} ${t.date} ${t.month} · ${t.hm}`;
    clockMeta.textContent =
      `Wind ${w.speed.toFixed(1)} m/s from ${compass(w.dirFrom)}${s.windOverride ? " (manual)" : ""} · barrage ${s.barrageOpen ? "open" : "closed"}`;
    $("speed-out").textContent = `${rate < 1 ? rate.toFixed(2) : rate.toFixed(1)} h/s${gpuLimited ? " · GPU max" : ""}`;
  };
  requestAnimationFrame(frame);
}

main().catch((e) => console.error(e));
