// Render stills of the 3D lake view headlessly (Deno WebGPU). Doubles as a
// shader/pipeline smoke test for web/src/lake3d.wgsl.
//
//   deno run --allow-read --allow-write --allow-env render/still3d.ts --data web/public/data/latest --out stills

import { parseArgs } from "jsr:@std/cli@1.0.17/parse-args";
import { encode } from "fast-png";
import { buildSimData, spinupHours, type WindForcing } from "../sim/data.ts";
import { HyacinthSim, type ShaderSources } from "../sim/engine.ts";
import { Lake3D } from "../web/src/lake3d.ts";

const args = parseArgs(Deno.args, { string: ["data", "out", "hours"], default: { data: "web/public/data/latest", out: "stills", hours: "30" } });
const here = new URL(".", import.meta.url);
const read = (p: string) => Deno.readTextFile(new URL(p, here));
const shaders: ShaderSources = {
  common: await read("../sim/shaders/common.wgsl"),
  lbm: await read("../sim/shaders/lbm.wgsl"),
  particles: await read("../sim/shaders/particles.wgsl"),
  accumulate: await read("../sim/shaders/accumulate.wgsl"),
  render: await read("../sim/shaders/render.wgsl"),
};
const lakeSrc = await read("../web/src/lake3d.wgsl");

const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
if (!adapter) throw new Error("no GPU adapter");
const device = await adapter.requestDevice({
  requiredLimits: {
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
  },
});
device.addEventListener("uncapturederror", (e) => console.error("GPU error:", (e as GPUUncapturedErrorEvent).error.message));

const wind = JSON.parse(await Deno.readTextFile(`${args.data}/wind.json`)) as WindForcing;
const data = buildSimData(await Deno.readFile(`${args.data}/mask.png`), await Deno.readFile(`${args.data}/seed.png`), wind);
const sim = await HyacinthSim.create(device, data, shaders);

// Run some hours so currents develop and jams accumulate.
const hours = Number(args.hours);
const spin = spinupHours(wind);
const t = performance.now();
const total = Math.round(hours * sim.stepsPerHour);
for (let done = 0; done < total; done += 200) {
  sim.advance(Math.min(200, total - done));
  if (done === 0 || sim.simHours < spin) sim.clearJam();
  await device.queue.onSubmittedWorkDone();
}
console.log(`simulated ${hours} h in ${((performance.now() - t) / 1000).toFixed(1)} s`);

const format: GPUTextureFormat = "rgba8unorm";
device.pushErrorScope("validation");
const lake = await Lake3D.create(sim, null, shaders.common, lakeSrc, format);
const err = await device.popErrorScope();
if (err) throw new Error(`3D pipeline validation: ${err.message}`);

const W = 1600;
const H = 1000;
const tex = device.createTexture({ size: [W, H], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
const bytesPerRow = Math.ceil((W * 4) / 256) * 256;
const buf = device.createBuffer({ size: bytesPerRow * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

// Find a busy spot (most mats) for the close-ups.
const parts = await sim.readParticles();
const grid = new Map<string, number>();
for (let i = 0; i < parts.length; i += 4) {
  if (parts[i + 2] < 0.5) continue;
  const k = `${Math.floor(parts[i] / 30)},${Math.floor(parts[i + 1] / 30)}`;
  grid.set(k, (grid.get(k) ?? 0) + 1);
}
const [busy] = [...grid.entries()].sort((a, b) => b[1] - a[1]);
const [bx, bz] = busy[0].split(",").map((v) => (Number(v) + 0.5) * 30);
console.log(`busiest spot ~(${bx}, ${bz}) with ${busy[1]} mats`);

const span = Math.max(sim.width, sim.height);
const w = sim.currentWind();
const shots: Array<{ name: string; hour: number; view: Parameters<Lake3D["look"]>[0] }> = [
  { name: "overview-day", hour: 10.5, view: { yaw: 0.35, pitch: 0.62, dist: span * 0.78, x: sim.width / 2, z: sim.height * 0.55 } },
  { name: "closeup-day", hour: 15, view: { yaw: 0.9, pitch: 0.42, dist: 70, x: bx, z: bz } },
  { name: "closeup-low", hour: 8, view: { yaw: 2.2, pitch: 0.2, dist: 28, x: bx, z: bz } },
  { name: "mid", hour: 9.5, view: { yaw: 0.6, pitch: 0.55, dist: 300, x: bx, z: bz } },
  { name: "sunset", hour: 17.85, view: { yaw: 1.7, pitch: 0.3, dist: 260, x: bx, z: bz } },
  { name: "night", hour: 22, view: { yaw: 0.35, pitch: 0.7, dist: span * 0.6, x: sim.width / 2, z: sim.height * 0.55 } },
];
await Deno.mkdir(args.out, { recursive: true });
for (const s of shots) {
  lake.look(s.view);
  device.pushErrorScope("validation");
  lake.renderTo(tex.createView(), W, H, { hour: s.hour, wind: { u: w.u, v: w.v, speed: w.speed } });
  const e = await device.popErrorScope();
  if (e) throw new Error(`${s.name}: ${e.message}`);
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow }, [W, H]);
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const raw = new Uint8Array(buf.getMappedRange());
  const px = new Uint8Array(W * H * 4);
  for (let y = 0; y < H; y++) px.set(raw.subarray(y * bytesPerRow, y * bytesPerRow + W * 4), y * W * 4);
  buf.unmap();
  await Deno.writeFile(`${args.out}/${s.name}.png`, encode({ width: W, height: H, data: px, channels: 4 }));
  console.log(`wrote ${args.out}/${s.name}.png`);
}
