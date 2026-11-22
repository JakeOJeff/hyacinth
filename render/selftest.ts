// Physics self-test for the shared WGSL simulation. Runs on the real GPU via
// Deno WebGPU, with small hand-built domains and known answers.
//
//   deno task selftest

import type { SimData, WindForcing } from "../sim/data.ts";
import { HyacinthSim, type ShaderSources } from "../sim/engine.ts";

const shaderDir = new URL("../sim/shaders/", import.meta.url);
const rd = (n: string) => Deno.readTextFile(new URL(n, shaderDir));
const shaders: ShaderSources = {
  common: await rd("common.wgsl"),
  lbm: await rd("lbm.wgsl"),
  particles: await rd("particles.wgsl"),
  accumulate: await rd("accumulate.wgsl"),
  render: await rd("render.wgsl"),
};

const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
if (!adapter) throw new Error("no WebGPU adapter");
const device = await adapter.requestDevice();
device.addEventListener("uncapturederror", (e) => {
  console.error("WebGPU error:", (e as GPUUncapturedErrorEvent).error.message);
  Deno.exit(1);
});
console.log(`adapter: ${adapter.info?.description || adapter.info?.vendor}`);

const DX = 40;
const DT = 10;
const BASE = { dtSeconds: DT, dropSlots: 64, maxParticles: 100_000, jitterMeters: 0, cohesion: 0 };

function wind(u: number, v: number, hours = 10): WindForcing {
  return {
    start: "2026-01-01T00:00:00Z", forecast_start: "2026-01-01T00:00:00Z", forecast_hours: hours,
    step_hours: 1, u: Array(hours + 1).fill(u), v: Array(hours + 1).fill(v),
    dx_m: DX, barrage_open: false, source: "selftest", synthetic: true,
  };
}

function domain(w: number, h: number, type: (x: number, y: number) => number, wf: WindForcing, seed?: (x: number, y: number) => number): SimData {
  const cellType = new Uint32Array(w * h);
  const s = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      cellType[y * w + x] = type(x, y);
      s[y * w + x] = seed ? seed(x, y) : 0;
    }
  }
  return { width: w, height: h, cellType, seed: s, wind: wf };
}

let failures = 0;
function check(name: string, ok: boolean, detail: string) {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}: ${detail}`);
  if (!ok) failures++;
}

function maxSpeed(vel: Float32Array): number {
  let m = 0;
  for (let i = 0; i < vel.length; i += 2) {
    const s = Math.hypot(vel[i], vel[i + 1]);
    if (!Number.isFinite(s)) return Infinity;
    m = Math.max(m, s);
  }
  return m;
}

// 1. Closed basin, no wind: fluid stays at rest, mass conserved.
{
  const sim = await HyacinthSim.create(device, domain(64, 64, () => 1, wind(0, 0)), shaders, BASE);
  const m0 = (await sim.readFlow()).mass;
  sim.advance(3000);
  const { vel, mass } = await sim.readFlow();
  check("rest basin stays at rest", maxSpeed(vel) < 1e-6, `max |u| = ${maxSpeed(vel).toExponential(2)}`);
  check("rest basin conserves mass", Math.abs(mass - m0) / m0 < 1e-5, `drift ${(((mass - m0) / m0) * 100).toExponential(2)} %`);
  sim.destroy();
}

// 2. Closed basin with strong wind: stable, bounded, mass conserved.
{
  const sim = await HyacinthSim.create(device, domain(96, 64, () => 1, wind(15, 4)), shaders, BASE);
  const m0 = (await sim.readFlow()).mass;
  sim.advance(20000);
  const { vel, mass } = await sim.readFlow();
  const ms = maxSpeed(vel);
  check("windy basin stays stable", Number.isFinite(ms) && ms <= 0.12 + 1e-6, `max |u| = ${ms.toFixed(4)} (clamp 0.12)`);
  check("windy basin conserves mass", Math.abs(mass - m0) / m0 < 1e-3, `drift ${(((mass - m0) / m0) * 100).toFixed(4)} %`);
  sim.destroy();
}

// 3. Open channel (sea outlets at both ends, walls top/bottom), wind along it:
//    the core current should reach currentPerWind * W.
const W_SPEED = 6;
const channel = (x: number, _y: number, w: number) => (x === 0 || x === w - 1 ? 3 : 1);
{
  const w = 400;
  const h = 60;
  const sim = await HyacinthSim.create(device, domain(w, h, (x, y) => channel(x, y, w), wind(W_SPEED, 0)), shaders, BASE);
  sim.advance(8000); // 16 drag time constants (1 / 0.002 = 500 steps)
  const { vel } = await sim.readFlow();
  const c = (h / 2) * w + w / 2;
  const got = (vel[2 * c] * DX) / DT; // m/s
  const want = 0.02 * W_SPEED;
  check("wind-driven channel current", Math.abs(got - want) / want < 0.1,
    `${got.toFixed(4)} m/s at centre, expected ~${want.toFixed(3)} m/s (2 % of wind)`);
  check("no cross-channel drift", Math.abs(vel[2 * c + 1]) < 0.05 * Math.abs(vel[2 * c]),
    `v/u = ${(vel[2 * c + 1] / vel[2 * c]).toExponential(2)}`);

  // 4. A lone mat in that channel moves at current + 3 % windage.
  sim.dropHyacinth(100.5, h / 2 + 0.5, 0, 1);
  const steps = 600;
  const p0 = await sim.readParticles();
  const i0 = p0.findIndex((_, i) => i % 4 === 2 && p0[i] > 0.5) - 2;
  const x0 = p0[i0];
  sim.advance(steps);
  const p1 = await sim.readParticles();
  const moved = ((p1[i0] - x0) * DX) / (steps * DT); // m/s
  const expect = got + 0.03 * W_SPEED;
  check("mat drift = current + windage", Math.abs(moved - expect) / expect < 0.05,
    `${moved.toFixed(4)} m/s, expected ${expect.toFixed(4)} m/s`);
  sim.destroy();
}

// 5. Barrage: mats pushed north must not pass a closed barrage, and do pass an open one.
for (const open of [false, true]) {
  const w = 60;
  const h = 200;
  const wf = wind(0, 8); // wind toward north
  wf.barrage_open = open;
  const type = (_x: number, y: number) => (y === 0 ? 3 : y === 100 ? 4 : 1);
  const seed = (_x: number, y: number) => (y > 150 && y < 190 ? 255 : 0);
  const sim = await HyacinthSim.create(device, domain(w, h, type, wf, seed), shaders, BASE);
  sim.advance(30000);
  const p = await sim.readParticles();
  let north = 0;
  let alive = 0;
  let flushed = 0;
  for (let i = 0; i < p.length; i += 4) {
    if (p[i + 2] > 0.5) {
      alive++;
      if (p[i + 1] < 100) north++;
    } else if (p[i + 3] > 0) flushed++;
  }
  if (open) {
    check("open barrage lets mats through", north + flushed > 0.5 * sim.seededCount,
      `${north} north of barrage, ${flushed} flushed to sea, of ${sim.seededCount}`);
  } else {
    check("closed barrage holds mats", north === 0 && flushed === 0 && alive === sim.seededCount,
      `${north} crossed, ${flushed} flushed, ${alive}/${sim.seededCount} alive`);
  }
  sim.destroy();
}

// 6. Jam accounting: mats driven against a bank register a jam with an ETA.
{
  const w = 80;
  const h = 80;
  const type = (x: number) => (x >= 70 ? 0 : 1); // land on the east side
  const seed = (x: number, y: number) => (x > 10 && x < 30 && y > 20 && y < 60 ? 255 : 0);
  const sim = await HyacinthSim.create(device, domain(w, h, type, wind(8, 0, 30), seed), shaders, BASE);
  sim.clearJam();
  sim.advance(sim.stepsPerHour * 30);
  const { eta } = await sim.readJam();
  let jammed = 0;
  let maxX = 0;
  let minX = w;
  for (let i = 0; i < eta.length; i++) {
    if (eta[i] !== 0xffffffff) {
      jammed++;
      maxX = Math.max(maxX, i % w);
      minX = Math.min(minX, i % w);
    }
  }
  check("mats pile up against the windward bank", jammed > 20 && minX >= 66 && maxX <= 69,
    `${jammed} jammed cells in columns ${minX}..${maxX} (bank at 70, shore band 3 cells)`);
  sim.destroy();
}

// 7. River head with no wind drives flow from the inflow to the sea outlet.
{
  const w = 300;
  const h = 40;
  const type = (x: number) => (x === 0 ? 2 : x === w - 1 ? 3 : 1);
  const sim = await HyacinthSim.create(device, domain(w, h, type, wind(0, 0)), shaders, BASE);
  sim.advance(20000);
  const { vel } = await sim.readFlow();
  const c = (h / 2) * w + w / 2;
  const u = (vel[2 * c] * DX) / DT;
  check("river head drives flow toward the sea", u > 0.005 && Math.abs(vel[2 * c + 1]) < 0.05 * vel[2 * c],
    `${u.toFixed(4)} m/s downstream at mid-channel`);
  sim.destroy();
}

device.destroy();
if (failures) {
  console.error(`${failures} check(s) failed`);
  Deno.exit(1);
}
console.log("all checks passed");
