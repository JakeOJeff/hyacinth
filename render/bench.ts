// Steps per second on a dataset: deno run --allow-read render/bench.ts data/latest
import { buildSimData } from "../sim/data.ts";
import { HyacinthSim, type ShaderSources } from "../sim/engine.ts";

const dir = Deno.args[0] ?? "data/latest";
const sh = (n: string) => Deno.readTextFile(new URL(`../sim/shaders/${n}.wgsl`, import.meta.url));
const shaders: ShaderSources = {
  common: await sh("common"), lbm: await sh("lbm"), particles: await sh("particles"),
  accumulate: await sh("accumulate"), render: await sh("render"),
};
const wind = JSON.parse(await Deno.readTextFile(`${dir}/wind.json`));
const data = buildSimData(await Deno.readFile(`${dir}/mask.png`), await Deno.readFile(`${dir}/seed.png`), wind);
const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
if (!adapter) throw new Error("no adapter");
const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
const sim = await HyacinthSim.create(device, data, shaders);
const tiles = Math.ceil(sim.width / 8) * Math.ceil(sim.height / 8);
console.log(`grid ${sim.width}x${sim.height}, ${sim.activeTiles}/${tiles} tiles have water (${((100 * sim.activeTiles) / tiles).toFixed(0)} %), ${sim.seedParticles} mats`);
for (const [label, opts] of [["full step", {}], ["water only", { particles: false }]] as const) {
  sim.advance(200, opts);
  await device.queue.onSubmittedWorkDone();
  const n = 4000;
  const t0 = performance.now();
  sim.advance(n, opts);
  await device.queue.onSubmittedWorkDone();
  const ms = (performance.now() - t0) / n;
  console.log(`${label}: ${(ms * 1000).toFixed(0)} us/step, ${(1000 / ms / sim.stepsPerHour).toFixed(1)} sim-hours per wall second`);
}
