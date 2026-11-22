// Lake health: the multi-year satellite dataset (Sentinel-2 passes per zone +
// geostationary daily conditions), its dashboard, and the assistant's tools.

export type Metric = "health" | "hyacinth_pct" | "debris_ha_per_km2" | "chlorophyll" | "turbidity";
type Num = number | null;

export interface HealthZone {
  id: number;
  name: string;
  name_ml: string | null;
  water_ha: number;
  x: number | null;
  y: number | null;
}

interface Sighting {
  date: string;
  zone: number;
  ha: number;
  lon: number;
  lat: number;
  x: number;
  y: number;
}

export interface HealthData {
  generated: string;
  period: [string, string];
  sources: Record<string, string>;
  caveats: string[];
  metrics: Record<Metric, { label: string; unit: string; higher_is: "better" | "worse"; desc: string }>;
  zones: HealthZone[];
  passes: { date: string[]; time: string[]; clear: number[] };
  series: Record<Metric, Num[][]>;
  hotspots: { zone: number; passes_observed: number; debris_sightings: number }[];
  sightings: Sighting[];
  daily: { date: string[]; cloudiness: Num[]; insolation_kwh: Num[]; rain_mm: Num[]; tmax_c: Num[] };
  now: {
    time: string;
    ghi_wm2: number;
    clear_sky_wm2: number;
    daylight: boolean;
    cloudiness_latest_daylight: Num;
    cloudiness_today: Num;
    source: string;
  } | null;
  latest_pass: string | null;
  images: { hotspots: string; latest: string | null };
}

export const METRICS: Metric[] = ["health", "hyacinth_pct", "debris_ha_per_km2", "chlorophyll", "turbidity"];
const DAY = 86_400_000;
const SVG = "http://www.w3.org/2000/svg";
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export async function loadHealth(): Promise<HealthData | null> {
  try {
    const r = await fetch("./data/health/health.json");
    if (!r.ok || !(r.headers.get("content-type") ?? "").includes("json")) return null;
    const d = (await r.json()) as HealthData;
    return d.passes?.date?.length ? d : null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------- queries

const t = (iso: string) => Date.parse(iso.length === 10 ? `${iso}T00:00:00Z` : iso);
const median = (v: number[]) => {
  if (!v.length) return null;
  const s = [...v].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const round = (v: Num, d = 2) => (v === null ? null : Math.round(v * 10 ** d) / 10 ** d);
const daysAgo = (iso: string) => Math.max(0, Math.round((Date.now() - t(iso)) / DAY));

export function findZone(h: HealthData, name: string): number | null {
  const q = name.trim().toLowerCase();
  if (!q || ["whole lake", "lake", "all", "vembanad", "everywhere"].includes(q)) return 0;
  let i = h.zones.findIndex((z) => z.name.toLowerCase() === q || (z.name_ml ?? "") === name.trim());
  if (i < 0) i = h.zones.findIndex((z) => z.name.toLowerCase().includes(q) || q.includes(z.name.toLowerCase()));
  return i < 0 ? null : i;
}

function zoneOrThrow(h: HealthData, name: string): number {
  const z = findZone(h, name);
  if (z === null) throw new Error(`unknown zone "${name}". Zones include: ${h.zones.slice(0, 25).map((x) => x.name).join(", ")}…`);
  return z;
}

function dateArg(h: HealthData, s: string): number {
  return s === "latest" || !s ? t(h.passes.date[h.passes.date.length - 1]) + DAY : t(s);
}

/** Index of the pass nearest `when` (ms) with a value for (metric, zone). */
function nearestPass(h: HealthData, metric: Metric, zone: number, when: number): number | null {
  let best: number | null = null;
  let bd = Infinity;
  h.passes.date.forEach((d, i) => {
    if (h.series[metric][zone][i] === null) return;
    const dd = Math.abs(t(d) - when);
    if (dd < bd) {
      bd = dd;
      best = i;
    }
  });
  return best;
}

function monthUsual(h: HealthData, metric: Metric, zone: number, month: number): Num {
  const v = h.passes.date.map((d, i) => (new Date(t(d)).getUTCMonth() === month ? h.series[metric][zone][i] : null)).filter(
    (x): x is number => x !== null,
  );
  return median(v);
}

function seriesBetween(h: HealthData, metric: Metric, zone: number, a: number, b: number) {
  return h.passes.date
    .map((d, i) => ({ date: d, value: h.series[metric][zone][i], clear: h.passes.clear[i] }))
    .filter((p) => p.value !== null && t(p.date) >= a && t(p.date) <= b) as { date: string; value: number; clear: number }[];
}

function recurrence(h: HealthData, s: Sighting): number {
  // Same spot = within ~100 m (2.5 cells of 40 m) on any pass.
  const dates = new Set<string>();
  for (const o of h.sightings) if (Math.hypot(o.x - s.x, o.y - s.y) <= 2.5) dates.add(o.date);
  return dates.size;
}

export function healthTools(h: HealthData, ui: { showLayer: (l: "hotspots" | "latest" | "forecast") => void }) {
  const zoneName = (z: number) => h.zones[z].name;
  return {
    lake_health(a: Record<string, unknown>) {
      const zone = zoneOrThrow(h, String(a.zone ?? "Whole lake"));
      const when = dateArg(h, String(a.date ?? "latest"));
      const i = nearestPass(h, "health", zone, when) ?? nearestPass(h, "hyacinth_pct", zone, when);
      if (i === null) return { zone: zoneName(zone), error: "no clear pass with data for this zone" };
      const month = new Date(t(h.passes.date[i])).getUTCMonth();
      const prev = (m: Metric) => {
        for (let k = i - 1; k >= 0; k--) if (h.series[m][zone][k] !== null) return { date: h.passes.date[k], value: h.series[m][zone][k] };
        return null;
      };
      const indicators = Object.fromEntries(
        METRICS.map((m) => [m, {
          value: h.series[m][zone][i],
          usual_for_month: round(monthUsual(h, m, zone, month), 3),
          previous_pass: prev(m),
          unit: h.metrics[m].unit,
          higher_is: h.metrics[m].higher_is,
        }]),
      );
      return {
        zone: zoneName(zone),
        pass_date: h.passes.date[i],
        days_ago: daysAgo(h.passes.date[i]),
        water_visible_lake_wide: h.passes.clear[i],
        indicators,
        note: "Health index ranks this zone against its own record since Jan 2025 (100 = best seen); relative, not a water-quality standard.",
      };
    },

    metric_series(a: Record<string, unknown>) {
      const metric = (METRICS.includes(a.metric as Metric) ? a.metric : "health") as Metric;
      const zone = zoneOrThrow(h, String(a.zone ?? "Whole lake"));
      const pts = seriesBetween(h, metric, zone, t(String(a.start ?? h.period[0])), dateArg(h, String(a.end ?? "latest")));
      const base = { metric, unit: h.metrics[metric].unit, zone: zoneName(zone) };
      if (a.by === "month") {
        const by = new Map<string, number[]>();
        for (const p of pts) by.set(p.date.slice(0, 7), [...(by.get(p.date.slice(0, 7)) ?? []), p.value]);
        return { ...base, months: [...by].map(([month, v]) => ({ month, median: round(median(v), 3), passes: v.length })) };
      }
      const step = Math.max(1, Math.ceil(pts.length / 80));
      return { ...base, passes: pts.filter((_, k) => k % step === 0).map((p) => ({ date: p.date, value: p.value })), total_passes: pts.length };
    },

    rank_zones(a: Record<string, unknown>) {
      const metric = (METRICS.includes(a.metric as Metric) ? a.metric : "health") as Metric;
      const top = Math.min(20, Math.max(1, Number(a.top) || 8));
      const last = t(h.passes.date[h.passes.date.length - 1]);
      const rows = h.zones.slice(1).map((z) => {
        const v = h.series[metric][z.id];
        let value: Num;
        if (a.period === "latest") {
          let k = v.length - 1;
          while (k >= 0 && (v[k] === null || last - t(h.passes.date[k]) > 30 * DAY)) k--;
          value = k >= 0 ? v[k] : null;
        } else {
          const vals = v.filter((x): x is number => x !== null);
          value = vals.length >= 3 ? vals.reduce((s, x) => s + x, 0) / vals.length : null;
        }
        return { zone: z.name, zone_ml: z.name_ml, value: round(value, 3), water_ha: z.water_ha };
      }).filter((r) => r.value !== null) as { zone: string; zone_ml: string | null; value: number; water_ha: number }[];
      const worstFirst = h.metrics[metric].higher_is === "better" ? (x: number, y: number) => x - y : (x: number, y: number) => y - x;
      rows.sort((p, q) => worstFirst(p.value, q.value));
      return { metric, unit: h.metrics[metric].unit, period: a.period === "latest" ? "latest clear pass (within 30 days)" : `average ${h.period[0]} to now`, order: "worst first", zones: rows.slice(0, top) };
    },

    debris_sightings(a: Record<string, unknown>) {
      const from = t(String(a.start ?? h.period[0]));
      const to = dateArg(h, String(a.end ?? "latest"));
      const zone = a.zone ? findZone(h, String(a.zone)) : 0;
      const list = h.sightings.filter((s) => t(s.date) >= from && t(s.date) <= to && (!zone || s.zone === zone));
      list.sort((p, q) => q.date.localeCompare(p.date) || q.ha - p.ha);
      return {
        count: list.length,
        latest_pass: h.latest_pass,
        latest_pass_days_ago: h.latest_pass ? daysAgo(h.latest_pass) : null,
        sightings: list.slice(0, 30).map((s) => ({
          date: s.date, place: zoneName(s.zone), area_ha: s.ha, lat: s.lat, lon: s.lon, seen_on_passes: recurrence(h, s),
        })),
        confidence: "Low per sighting (young hyacinth clumps can look similar at 20 m); spots seen on several passes are more reliable. Positions are as of each pass, not live.",
      };
    },

    sky_now() {
      const n = h.daily.date.length;
      const last7 = h.daily.date.slice(n - 7).map((d, k) => ({
        date: d,
        cloudiness: h.daily.cloudiness[n - 7 + k],
        sunshine_kwh_m2: h.daily.insolation_kwh[n - 7 + k],
        rain_mm: h.daily.rain_mm[n - 7 + k],
      }));
      return {
        now: h.now,
        last_7_days: last7,
        last_clear_pass: h.latest_pass,
        last_clear_pass_days_ago: h.latest_pass ? daysAgo(h.latest_pass) : null,
        note: "Cloudiness and sunshine are measured by geostationary satellites every 15 min; rain is model data. Sentinel-2 passes every 2-5 days and needs a clear sky.",
      };
    },

    show_map_layer(a: Record<string, unknown>) {
      const layer = a.layer === "latest" || a.layer === "forecast" ? a.layer : "hotspots";
      ui.showLayer(layer);
      return { shown: layer };
    },
  };
}

// ---------------------------------------------------------------- dashboard

const $ = <T extends Element = HTMLElement>(id: string) => document.getElementById(id) as unknown as T;
const el = <K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};
const svgEl = <K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}) => {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
};
const fmtDate = (iso: string) => {
  const d = new Date(t(iso));
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
};

function fmt(m: Metric, v: number): string {
  if (m === "health") return `${Math.round(v)}`;
  if (m === "hyacinth_pct") return `${v.toFixed(1)}%`;
  if (m === "debris_ha_per_km2") return v.toFixed(2);
  return v.toFixed(3);
}

function niceTicks(lo: number, hi: number, n = 4): number[] {
  if (hi - lo < 1e-9) hi = lo + 1;
  const raw = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(Math.round(v / step) * step);
  return out;
}

interface Tip {
  show: (x: number, y: number, strong: string, lines: string[]) => void;
  hide: () => void;
}

function tooltip(host: HTMLElement): Tip {
  const tip = el("div", "tip");
  tip.hidden = true;
  host.append(tip);
  return {
    show(x, y, strong, lines) {
      tip.replaceChildren(el("b", "", strong), ...lines.map((l) => el("span", "", l)));
      tip.hidden = false;
      const w = host.clientWidth;
      tip.style.left = `${Math.min(Math.max(x, 70), w - 70)}px`;
      tip.style.top = `${y}px`;
    },
    hide() {
      tip.hidden = true;
    },
  };
}

export function renderHealth(h: HealthData, ui: { showZone: (zone: HealthZone) => void; showLayer: (l: "hotspots" | "latest" | "forecast") => void }) {
  $("health").hidden = false;
  for (const e of document.querySelectorAll<HTMLElement>("[data-needs-health]")) e.hidden = false;
  let metric: Metric = "health";
  let zone = 0;

  // ---- summary tiles ----
  const lastI = h.passes.date.length - 1;
  const li = nearestPass(h, "health", 0, Date.now());
  if (li !== null) {
    const v = h.series.health[0][li]!;
    const usual = monthUsual(h, "health", 0, new Date(t(h.passes.date[li])).getUTCMonth());
    $("t-health").textContent = `${Math.round(v)}`;
    $("t-health-sub").textContent = usual === null
      ? `pass ${fmtDate(h.passes.date[li])}`
      : `${v >= usual ? "▲ better" : "▼ worse"} than usual for ${MONTHS[new Date(t(h.passes.date[li])).getUTCMonth()]}`;
  }
  if (h.latest_pass) {
    const ago = daysAgo(h.latest_pass);
    $("t-pass").textContent = fmtDate(h.latest_pass).replace(/ \d{4}$/, "");
    $("t-pass-sub").textContent = `${ago === 0 ? "today" : `${ago} day${ago === 1 ? "" : "s"} ago`} · ${Math.round(h.passes.clear[lastI] * 100)}% of water visible`;
    const n = h.sightings.filter((s) => s.date === h.latest_pass).length;
    $("t-debris").textContent = String(n);
    $("t-debris-sub").textContent = `possible debris patches, ${fmtDate(h.latest_pass).replace(/ \d{4}$/, "")}`;
  }
  if (h.now) {
    const c = h.now.daylight ? h.now.cloudiness_today ?? h.now.cloudiness_latest_daylight : h.now.cloudiness_latest_daylight;
    const word = c === null ? "–" : c < 0.2 ? "Clear" : c < 0.5 ? "Partly cloudy" : c < 0.8 ? "Cloudy" : "Overcast";
    $("t-sky").textContent = h.now.daylight ? word : `Night`;
    const tt = new Date(t(h.now.time) + 5.5 * 3600_000);
    const hm = `${String(tt.getUTCHours()).padStart(2, "0")}:${String(tt.getUTCMinutes()).padStart(2, "0")} IST`;
    $("t-sky-sub").textContent = h.now.daylight
      ? `${c === null ? "" : `${Math.round(c * 100)}% cloud today · `}${h.now.ghi_wm2} W/m² at ${hm}`
      : `${c === null ? "" : `${Math.round(c * 100)}% cloud at last light · `}reading ${hm}`;
  }

  // ---- controls ----
  const sel = $<HTMLSelectElement>("h-zone");
  sel.replaceChildren();
  h.zones.forEach((z, i) => {
    const o = el("option", "", z.name_ml && i ? `${z.name} · ${z.name_ml}` : z.name);
    o.value = String(i);
    sel.append(o);
  });
  const metricBox = $("h-metrics");
  metricBox.replaceChildren();
  for (const m of METRICS) {
    const b = el("button", "", h.metrics[m].label.replace(" (chlorophyll proxy)", ""));
    b.dataset.metric = m;
    b.setAttribute("role", "tab");
    b.addEventListener("click", () => {
      metric = m;
      draw();
    });
    metricBox.append(b);
  }
  sel.addEventListener("change", () => {
    zone = Number(sel.value);
    draw();
  });
  $("h-show").addEventListener("click", () => {
    if (zone) ui.showZone(h.zones[zone]);
    else ui.showLayer("hotspots");
  });
  $("h-table-toggle").addEventListener("click", () => {
    const tbl = $("h-table");
    tbl.hidden = !tbl.hidden;
    $("h-table-toggle").textContent = tbl.hidden ? "Table" : "Chart only";
  });
  $("h-csv").addEventListener("click", () => {
    const rows = [["date", "zone", metric, "lake_water_visible"]];
    h.passes.date.forEach((d, i) => {
      const v = h.series[metric][zone][i];
      if (v !== null) rows.push([d, h.zones[zone].name, String(v), String(h.passes.clear[i])]);
    });
    const blob = new Blob([rows.map((r) => r.map((c) => (/[",]/.test(c) ? `"${c.replace(/"/g, '""')}"` : c)).join(",")).join("\n")], { type: "text/csv" });
    const a = el("a");
    a.href = URL.createObjectURL(blob);
    a.download = `hyacinth-${metric}-${h.zones[zone].name.replace(/\W+/g, "-").toLowerCase()}.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  });
  $("h-period").textContent = `${fmtDate(h.period[0])} – ${fmtDate(h.passes.date[lastI])} · ${h.passes.date.length} clear passes · ${h.zones.length - 1} zones`;
  const caveats = $("h-caveats");
  caveats.replaceChildren(...h.caveats.map((c) => el("li", "", c)));
  const sources = $("h-sources");
  sources.replaceChildren(...Object.values(h.sources).map((c) => el("li", "", c)));

  renderSky(h);
  renderDebris(h, ui);

  const draw = () => {
    for (const b of metricBox.querySelectorAll<HTMLButtonElement>("button")) b.setAttribute("aria-selected", String(b.dataset.metric === metric));
    const info = h.metrics[metric];
    $("h-desc").textContent = info.desc;
    $("h-unit").textContent = info.unit;
    renderTrend(h, metric, zone);
    renderSeason(h, metric, zone);
    renderRanking(h, metric, (z) => {
      zone = z;
      sel.value = String(z);
      draw();
    });
  };
  draw();
  new ResizeObserver(() => {
    renderTrend(h, metric, zone);
    renderSeason(h, metric, zone);
    renderSky(h);
  }).observe($("h-trend-wrap"));
}

// ---- trend: one series, crosshair + tooltip, table view ----
function renderTrend(h: HealthData, metric: Metric, zone: number) {
  const wrap = $("h-trend-wrap");
  const svg = $<SVGSVGElement>("h-trend");
  svg.replaceChildren();
  wrap.querySelector(".tip")?.remove();
  const W = Math.max(300, wrap.clientWidth);
  const H = 260;
  const m = { l: 44, r: 14, t: 14, b: 28 };
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const pts = h.passes.date
    .map((d, i) => ({ d, x: t(d), v: h.series[metric][zone][i], c: h.passes.clear[i] }))
    .filter((p) => p.v !== null) as { d: string; x: number; v: number; c: number }[];
  const empty = $("h-empty");
  empty.hidden = pts.length > 1;
  if (pts.length < 2) return;
  const x0 = t(h.period[0]);
  const x1 = Math.max(t(h.passes.date[h.passes.date.length - 1]), x0 + DAY);
  let lo = Math.min(...pts.map((p) => p.v));
  let hi = Math.max(...pts.map((p) => p.v));
  if (metric === "health") [lo, hi] = [0, 100];
  else if (metric === "hyacinth_pct" || metric === "debris_ha_per_km2") lo = 0;
  const pad = (hi - lo) * 0.08 || 1;
  const ticks = niceTicks(lo, hi + (metric === "health" ? 0 : pad));
  const ylo = Math.min(lo, ticks[0]);
  const yhi = Math.max(hi, ticks[ticks.length - 1]);
  const X = (v: number) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
  const Y = (v: number) => H - m.b - ((v - ylo) / (yhi - ylo)) * (H - m.t - m.b);

  const g = svgEl("g", { class: "axis" });
  for (const v of ticks) {
    g.append(svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), class: "grid" }));
    const lab = svgEl("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" });
    lab.textContent = fmt(metric, v).replace(/\.0+%$/, "%");
    g.append(lab);
  }
  // Month labels: every 3rd month, year on January.
  const start = new Date(x0);
  for (let k = 0; ; k += 3) {
    const d = Date.UTC(start.getUTCFullYear(), start.getUTCMonth() + k, 1);
    if (d > x1) break;
    const dd = new Date(d);
    const lab = svgEl("text", { x: X(d), y: H - 8, "text-anchor": "middle" });
    lab.textContent = dd.getUTCMonth() === 0 || k === 0 ? `${MONTHS[dd.getUTCMonth()]} ${String(dd.getUTCFullYear()).slice(2)}` : MONTHS[dd.getUTCMonth()];
    g.append(lab);
  }
  svg.append(g);

  // Monsoon-style gaps (> 40 days between passes) break the line.
  let d = "";
  pts.forEach((p, i) => {
    const gap = i > 0 && p.x - pts[i - 1].x > 40 * DAY;
    d += `${i === 0 || gap ? "M" : "L"}${X(p.x).toFixed(1)},${Y(p.v).toFixed(1)}`;
  });
  svg.append(svgEl("path", { d, class: "line" }));
  for (const p of pts) svg.append(svgEl("circle", { cx: X(p.x), cy: Y(p.v), r: 3, class: "pt" }));
  const last = pts[pts.length - 1];
  const endLab = svgEl("text", { x: X(last.x) - 6, y: Y(last.v) - 10, "text-anchor": "end", class: "end" });
  endLab.textContent = fmt(metric, last.v);
  svg.append(endLab);

  // Crosshair snaps to the nearest pass.
  const cross = svgEl("line", { y1: m.t, y2: H - m.b, class: "cross", visibility: "hidden" });
  const hot = svgEl("circle", { r: 5, class: "hot", visibility: "hidden" });
  svg.append(cross, hot);
  const tip = tooltip(wrap);
  let cur = pts.length - 1;
  const show = (i: number) => {
    cur = i;
    const p = pts[i];
    cross.setAttribute("x1", String(X(p.x)));
    cross.setAttribute("x2", String(X(p.x)));
    cross.setAttribute("visibility", "visible");
    hot.setAttribute("cx", String(X(p.x)));
    hot.setAttribute("cy", String(Y(p.v)));
    hot.setAttribute("visibility", "visible");
    const r = svg.getBoundingClientRect();
    tip.show((X(p.x) / W) * r.width, (Y(p.v) / H) * r.height - 12, fmt(metric, p.v), [fmtDate(p.d), `${Math.round(p.c * 100)}% of lake visible`]);
  };
  const hide = () => {
    cross.setAttribute("visibility", "hidden");
    hot.setAttribute("visibility", "hidden");
    tip.hide();
  };
  svg.onpointermove = (e) => {
    const r = svg.getBoundingClientRect();
    const xv = x0 + (((e.clientX - r.left) / r.width) * W - m.l) / (W - m.l - m.r) * (x1 - x0);
    let best = 0;
    for (let i = 1; i < pts.length; i++) if (Math.abs(pts[i].x - xv) < Math.abs(pts[best].x - xv)) best = i;
    show(best);
  };
  svg.onpointerleave = hide;
  svg.onfocus = () => show(cur);
  svg.onblur = hide;
  svg.onkeydown = (e) => {
    if (e.key === "ArrowLeft") show(Math.max(0, cur - 1));
    else if (e.key === "ArrowRight") show(Math.min(pts.length - 1, cur + 1));
    else return;
    e.preventDefault();
  };
  svg.setAttribute("aria-label", `${h.metrics[metric].label} for ${h.zones[zone].name}, ${pts.length} satellite passes. Latest ${fmt(metric, last.v)} on ${fmtDate(last.d)}.`);

  // Table view (same numbers, no hover needed).
  const tbody = $("h-tbody");
  tbody.replaceChildren(
    ...[...pts].reverse().map((p) => {
      const tr = el("tr");
      tr.append(el("td", "", fmtDate(p.d)), el("td", "num", fmt(metric, p.v)), el("td", "num", `${Math.round(p.c * 100)}%`));
      return tr;
    }),
  );
}

// ---- seasonal pattern: monthly medians, columns ----
function renderSeason(h: HealthData, metric: Metric, zone: number) {
  const wrap = $("h-season-wrap");
  const svg = $<SVGSVGElement>("h-season");
  svg.replaceChildren();
  wrap.querySelector(".tip")?.remove();
  const W = Math.max(260, wrap.clientWidth);
  const H = 180;
  const m = { l: 8, r: 8, t: 22, b: 24 };
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const byMonth = MONTHS.map((_, k) => {
    const v = h.passes.date.map((d, i) => (new Date(t(d)).getUTCMonth() === k ? h.series[metric][zone][i] : null)).filter((x): x is number => x !== null);
    return { med: median(v), n: v.length };
  });
  const vals = byMonth.map((b) => b.med).filter((x): x is number => x !== null);
  if (!vals.length) return;
  const lo = metric === "health" || metric === "hyacinth_pct" || metric === "debris_ha_per_km2" ? 0 : Math.min(...vals);
  const hi = metric === "health" ? 100 : Math.max(...vals);
  const span = hi - lo || 1;
  const slot = (W - m.l - m.r) / 12;
  const bw = Math.min(24, slot * 0.6);
  const base = H - m.b;
  const tip = tooltip(wrap);
  const worst = h.metrics[metric].higher_is === "better" ? Math.min(...vals) : Math.max(...vals);
  byMonth.forEach((b, k) => {
    const cx = m.l + slot * (k + 0.5);
    const lab = svgEl("text", { x: cx, y: H - 6, "text-anchor": "middle", class: "axis-t" });
    lab.textContent = MONTHS[k].slice(0, 1);
    svg.append(lab);
    if (b.med === null) {
      svg.append(svgEl("line", { x1: cx - bw / 2, x2: cx + bw / 2, y1: base, y2: base, class: "none" }));
      return;
    }
    const hgt = Math.max(3, ((b.med - lo) / span) * (H - m.t - m.b));
    const x = cx - bw / 2;
    const y = base - hgt;
    const r = Math.min(4, hgt);
    const bar = svgEl("path", {
      d: `M${x},${base}V${y + r}Q${x},${y} ${x + r},${y}H${x + bw - r}Q${x + bw},${y} ${x + bw},${y + r}V${base}Z`,
      class: b.med === worst ? "col peak" : "col",
      tabindex: 0,
    });
    const showTip = () => {
      const rr = svg.getBoundingClientRect();
      tip.show((cx / W) * rr.width, (y / H) * rr.height - 8, fmt(metric, b.med!), [`${MONTHS[k]} · median of ${b.n} pass${b.n === 1 ? "" : "es"}`]);
    };
    bar.addEventListener("pointerenter", showTip);
    bar.addEventListener("focus", showTip);
    bar.addEventListener("pointerleave", () => tip.hide());
    bar.addEventListener("blur", () => tip.hide());
    svg.append(bar);
    if (b.med === worst) {
      const v = svgEl("text", { x: cx, y: y - 6, "text-anchor": "middle", class: "end" });
      v.textContent = fmt(metric, b.med);
      svg.append(v);
    }
  });
  $("h-season-note").textContent = `Darker column = ${h.metrics[metric].higher_is === "better" ? "worst (lowest)" : "worst (highest)"} month. Months without clear passes are flat.`;
}

// ---- ranking: zones worst first ----
function renderRanking(h: HealthData, metric: Metric, pick: (zone: number) => void) {
  const box = $("h-rank");
  box.replaceChildren();
  const rows = h.zones.slice(1).map((z) => {
    const v = h.series[metric][z.id].filter((x): x is number => x !== null);
    return { z, v: v.length >= 3 ? v.reduce((s, x) => s + x, 0) / v.length : null };
  }).filter((r) => r.v !== null) as { z: HealthZone; v: number }[];
  const better = h.metrics[metric].higher_is === "better";
  rows.sort((a, b) => (better ? a.v - b.v : b.v - a.v));
  const top = rows.slice(0, 8);
  const vals = rows.map((r) => r.v);
  const lo = metric === "health" ? 0 : Math.min(0, ...vals);
  const hi = metric === "health" ? 100 : Math.max(...vals);
  top.forEach((r, i) => {
    const b = el("button", "rank-row");
    b.style.setProperty("--i", String(i));
    const name = el("span", "rank-name", r.z.name);
    const track = el("span", "rank-track");
    const fill = el("i");
    fill.style.width = `${Math.max(4, ((r.v - lo) / (hi - lo || 1)) * 100)}%`;
    track.append(fill);
    b.append(el("span", "rank-n", String(i + 1)), name, track, el("span", "rank-v", fmt(metric, r.v)));
    b.title = `Average since ${fmtDate(h.period[0])}. Click to chart this zone.`;
    b.addEventListener("click", () => pick(r.z.id));
    box.append(b);
  });
  $("h-rank-note").textContent = `${better ? "Lowest" : "Highest"} average ${h.metrics[metric].label.toLowerCase()} since ${MONTHS[new Date(t(h.period[0])).getUTCMonth()]} ${new Date(t(h.period[0])).getUTCFullYear()}.`;
}

// ---- debris: latest pass sightings ----
function renderDebris(h: HealthData, ui: { showZone: (zone: HealthZone) => void; showLayer: (l: "hotspots" | "latest" | "forecast") => void }) {
  const box = $("h-debris");
  box.replaceChildren();
  const recent = h.latest_pass ? h.sightings.filter((s) => s.date === h.latest_pass) : [];
  recent.sort((a, b) => recurrence(h, b) - recurrence(h, a) || b.ha - a.ha);
  $("h-debris-head").textContent = h.latest_pass
    ? `${recent.length} possible patches on ${fmtDate(h.latest_pass)} (${daysAgo(h.latest_pass)} days ago)`
    : "No clear pass yet";
  for (const s of recent.slice(0, 6)) {
    const seen = recurrence(h, s);
    const row = el("li", "deb-row");
    const txt = el("div");
    txt.append(el("b", "", h.zones[s.zone]?.name ?? "Open lake"), el("span", "", `${s.ha.toFixed(2)} ha · seen on ${seen} pass${seen === 1 ? "" : "es"}`));
    const conf = el("span", `conf ${seen >= 3 ? "hi" : seen === 2 ? "mid" : "lo"}`, seen >= 3 ? "Recurring" : seen === 2 ? "Seen twice" : "Single");
    const btn = el("button", "chip-btn", "Show");
    btn.addEventListener("click", () => ui.showZone({ id: s.zone, name: h.zones[s.zone]?.name ?? "", name_ml: null, water_ha: 0, x: s.x, y: s.y }));
    row.append(txt, conf, btn);
    box.append(row);
  }
  $("h-layer-latest").onclick = () => ui.showLayer("latest");
  $("h-layer-hot").onclick = () => ui.showLayer("hotspots");
}

// ---- sky strip: geostationary cloudiness + rain per day, Sentinel passes on top ----
function renderSky(h: HealthData) {
  const wrap = $("h-sky-wrap");
  const svg = $<SVGSVGElement>("h-sky");
  svg.replaceChildren();
  wrap.querySelector(".tip")?.remove();
  const W = Math.max(300, wrap.clientWidth);
  const H = 104;
  const l = 92;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const n = h.daily.date.length;
  const cw = (W - l) / n;
  const rows = [
    { y: 18, label: "Sentinel-2 pass", kind: "pass" },
    { y: 40, label: "Cloud (geo sat)", kind: "cloud" },
    { y: 66, label: "Rain (model)", kind: "rain" },
  ] as const;
  const passes = new Set(h.passes.date);
  for (const r of rows) {
    const lab = svgEl("text", { x: 0, y: r.y + 12, class: "axis-t" });
    lab.textContent = r.label;
    svg.append(lab);
  }
  // Sequential single hue: light (low) -> dark slate (high).
  const ramp = (f: number) => {
    const a = [226, 232, 240];
    const b = [30, 58, 138];
    return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",")})`;
  };
  const g = svgEl("g");
  for (let i = 0; i < n; i++) {
    const x = l + i * cw;
    const d = h.daily.date[i];
    if (passes.has(d)) g.append(svgEl("rect", { x, y: rows[0].y + 4, width: Math.max(1.5, cw), height: 14, rx: 1, class: "pass" }));
    const c = h.daily.cloudiness[i];
    g.append(svgEl("rect", { x, y: rows[1].y, width: cw + 0.3, height: 20, fill: c === null ? "transparent" : ramp(c) }));
    const rr = h.daily.rain_mm[i];
    g.append(svgEl("rect", { x, y: rows[2].y, width: cw + 0.3, height: 20, fill: rr === null ? "transparent" : ramp(Math.min(1, Math.log1p(rr) / Math.log1p(60))) }));
  }
  svg.append(g);
  // Month ticks.
  for (let i = 0; i < n; i++) {
    if (!h.daily.date[i].endsWith("-01")) continue;
    const mo = Number(h.daily.date[i].slice(5, 7)) - 1;
    if (mo % 3) continue;
    const lab = svgEl("text", { x: l + i * cw, y: H - 2, class: "axis-t" });
    lab.textContent = mo === 0 ? `${MONTHS[mo]} ${h.daily.date[i].slice(2, 4)}` : MONTHS[mo];
    svg.append(lab);
  }
  const tip = tooltip(wrap);
  const cross = svgEl("line", { y1: rows[0].y, y2: rows[2].y + 20, class: "cross", visibility: "hidden" });
  svg.append(cross);
  svg.onpointermove = (e) => {
    const r = svg.getBoundingClientRect();
    const xv = ((e.clientX - r.left) / r.width) * W;
    const i = Math.floor((xv - l) / cw);
    if (i < 0 || i >= n) return tip.hide();
    cross.setAttribute("x1", String(l + (i + 0.5) * cw));
    cross.setAttribute("x2", String(l + (i + 0.5) * cw));
    cross.setAttribute("visibility", "visible");
    const c = h.daily.cloudiness[i];
    const s = h.daily.insolation_kwh[i];
    const rr = h.daily.rain_mm[i];
    tip.show(((l + (i + 0.5) * cw) / W) * r.width, (rows[1].y / H) * r.height - 6, c === null ? "No sat data" : `${Math.round(c * 100)}% cloud`, [
      fmtDate(h.daily.date[i]),
      `${s === null ? "–" : s.toFixed(1)} kWh/m² sun · ${rr === null ? "–" : rr.toFixed(1)} mm rain`,
      passes.has(h.daily.date[i]) ? "✓ clear Sentinel-2 pass" : "",
    ].filter(Boolean));
  };
  svg.onpointerleave = () => {
    tip.hide();
    cross.setAttribute("visibility", "hidden");
  };
}
