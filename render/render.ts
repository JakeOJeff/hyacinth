// Headless daily render (Deno + WebGPU). Same shaders and engine as the
// browser demo.
//
//   deno task render --data data/latest [--fps 30] [--seconds 15] [--frames-dir]
//
// Reads  <data>/mask.png, seed.png, wind.json (+ observed.png if present)
// Writes <data>/out/forecast.mp4    15 s timelapse of the forecast window
//                   heatmap.png     jam heatmap (dwell time)
//                   final.png       last live frame
//                   jam.bin         float32 [H*W] mat-hours per cell (dwell / jamThreshold)
//                   eta.bin         float32 [H*W] hours after forecast_start a cell
//                                   first jammed, -1 = never
//                   density.bin     float32 [H*W] particles per cell at the end
//                   state.bin       float32 vec4 per seeded mat at forecast start
//                                   (lets the browser skip the spin-up)
//                   stats.json      run metadata
//
// Set FFMPEG=/path/to/ffmpeg if it is not on PATH. Without ffmpeg the frames
// are written as PNGs into out/frames/ instead.

import { parseArgs } from "jsr:@std/cli@1.0.17/parse-args";
import { encode } from "fast-png";
import { buildSimData, spinupHours, type WindForcing } from "../sim/data.ts";
import { NO_ETA, HyacinthSim, type ShaderSources } from "../sim/engine.ts";

const args = parseArgs(Deno.args, {
  string: ["data", "out", "hours"],
  boolean: ["frames-dir", "help"],
  default: { fps: 30, seconds: 15 },
});
if (args.help || !args.data) {
  console.log("usage: deno task render --data <dir> [--out <dir>] [--fps 30] [--seconds 15] [--hours N] [--frames-dir]");
  Deno.exit(args.help ? 0 : 2);
}

const here = new URL(".", import.meta.url);
const shaderDir = new URL("../sim/shaders/", here);
const readShader = (name: string) => Deno.readTextFile(new URL(name, shaderDir));
const shaders: ShaderSources = {
  common: await readShader("common.wgsl"),
  lbm: await readShader("lbm.wgsl"),
  particles: await readShader("particles.wgsl"),
  accumulate: await readShader("accumulate.wgsl"),
  render: await readShader("render.wgsl"),
};

const dataDir = args.data.replace(/[\\/]+$/, "");
const outDir = (args.out ?? `${dataDir}/out`).replace(/[\\/]+$/, "");
await Deno.mkdir(outDir, { recursive: true });

const wind: WindForcing = JSON.parse(await Deno.readTextFile(`${dataDir}/wind.json`));
let observed: Uint8Array | undefined;
try {
  observed = await Deno.readFile(`${dataDir}/observed.png`);
} catch (e) {
  if (!(e instanceof Deno.errors.NotFound)) throw e;
}
const data = buildSimData(
  await Deno.readFile(`${dataDir}/mask.png`),
  await Deno.readFile(`${dataDir}/seed.png`),
  wind,
  observed,
);

const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
if (!adapter) throw new Error("no WebGPU adapter (on Linux CI install mesa-vulkan-drivers for lavapipe)");
const device = await adapter.requestDevice({
  requiredLimits: {
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
  },
});
device.addEventListener("uncapturederror", (e) => {
  console.error("WebGPU error:", (e as GPUUncapturedErrorEvent).error.message);
  Deno.exit(1);
});
console.log(`[render] adapter: ${adapter.info?.description || adapter.info?.vendor || "unknown"}`);

const sim = await HyacinthSim.create(device, data, shaders);
const W = data.width;
const H = data.height;
const fps = Number(args.fps);
const seconds = Number(args.seconds);
const forecastHours = args.hours ? Number(args.hours) : wind.forecast_hours;
const spinup = spinupHours(wind);
console.log(
  `[render] grid ${W}x${H} @ ${wind.dx_m} m, dt ${sim.dtSeconds.toFixed(2)} s, ${sim.stepsPerHour} steps/h, ` +
    `${sim.seededCount} mat particles (${sim.particlesPerSeedCell.toFixed(2)} per full cell), ` +
    `spin-up ${spinup.toFixed(1)} h, forecast ${forecastHours} h, barrage ${wind.barrage_open ? "open" : "closed"}`,
);
if (wind.synthetic) console.log("[render] NOTE: synthetic data, not a real forecast");

const t0 = performance.now();
async function advanceSync(steps: number) {
  const chunk = 2048;
  for (let done = 0; done < steps; done += chunk) {
    sim.advance(Math.min(chunk, steps - done));
    await device.queue.onSubmittedWorkDone();
  }
}

// Spin-up from the satellite scene to the forecast start.
const spinSteps = Math.round(spinup * sim.stepsPerHour);
if (spinSteps > 0) {
  await advanceSync(spinSteps);
  console.log(`[render] spin-up done in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
}
sim.clearJam();
const startParticles = await sim.particleStats();
const stateStep = sim.step;
await Deno.writeFile(`${outDir}/state.bin`, new Uint8Array((await sim.readSeededParticles()).buffer));

// Frame capture.
const format: GPUTextureFormat = "rgba8unorm";
const tex = device.createTexture({
  size: [W, H],
  format,
  usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
});
const bytesPerRow = Math.ceil((W * 4) / 256) * 256;
const readBuf = device.createBuffer({ size: bytesPerRow * H, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

async function grab(): Promise<Uint8Array> {
  const enc = device.createCommandEncoder();
  sim.render(tex.createView(), format, enc);
  enc.copyTextureToBuffer({ texture: tex }, { buffer: readBuf, bytesPerRow, rowsPerImage: H }, [W, H]);
  device.queue.submit([enc.finish()]);
  await readBuf.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(readBuf.getMappedRange());
  const out = new Uint8Array(W * H * 4);
  for (let y = 0; y < H; y++) out.set(src.subarray(y * bytesPerRow, y * bytesPerRow + W * 4), y * W * 4);
  readBuf.unmap();
  return out;
}

const savePng = (path: string, rgba: Uint8Array) =>
  Deno.writeFile(path, encode({ width: W, height: H, data: rgba, channels: 4, depth: 8 }));

// Video sink: ffmpeg over stdin, or PNG frames.
const ffmpegBin = Deno.env.get("FFMPEG") ?? "ffmpeg";
const scale = W < 600 ? 2 : 1;
let ffmpeg: Deno.ChildProcess | null = null;
let writer: WritableStreamDefaultWriter<Uint8Array> | null = null;
if (!args["frames-dir"]) {
  try {
    ffmpeg = new Deno.Command(ffmpegBin, {
      args: [
        "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", `${W}x${H}`, "-r", String(fps), "-i", "-",
        "-vf", `scale=trunc(iw*${scale}/2)*2:trunc(ih*${scale}/2)*2:flags=neighbor`,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
        `${outDir}/forecast.mp4`,
      ],
      stdin: "piped",
      stdout: "inherit",
      stderr: "inherit",
    }).spawn();
    writer = ffmpeg.stdin.getWriter();
  } catch (e) {
    if (!(e instanceof Deno.errors.NotFound)) throw e;
    console.warn(`[render] ${ffmpegBin} not found; writing PNG frames instead`);
  }
}
const framesDir = `${outDir}/frames`;
await Deno.remove(framesDir, { recursive: true }).catch(() => {});
if (!writer) await Deno.mkdir(framesDir, { recursive: true });

const totalSteps = Math.round(forecastHours * sim.stepsPerHour);
const frames = Math.max(1, Math.round(fps * seconds));
let doneSteps = 0;
for (let f = 0; f < frames; f++) {
  const target = Math.round(((f + 1) / frames) * totalSteps);
  await advanceSync(target - doneSteps);
  doneSteps = target;
  sim.progress = doneSteps / totalSteps;
  const rgba = await grab();
  if (writer) await writer.write(rgba);
  else await savePng(`${framesDir}/${String(f).padStart(4, "0")}.png`, rgba);
  if (f % Math.max(1, Math.floor(frames / 10)) === 0) {
    console.log(`[render] frame ${f + 1}/${frames}  T+${(doneSteps / sim.stepsPerHour).toFixed(1)} h`);
  }
}
if (writer && ffmpeg) {
  await writer.close();
  const st = await ffmpeg.status;
  if (!st.success) throw new Error(`ffmpeg exited with code ${st.code}`);
}

sim.progress = 0;
await savePng(`${outDir}/final.png`, await grab());
sim.viewMode = 1;
await savePng(`${outDir}/heatmap.png`, await grab());
sim.viewMode = 0;

// Numeric outputs for the caption pipeline.
const { jam, eta, density } = await sim.readJam();
const n = W * H;
const jamHours = new Float32Array(n);
const etaHours = new Float32Array(n);
const dens = new Float32Array(n);
const perJamHour = sim.jamThreshold * sim.stepsPerHour;
let jammedCells = 0;
for (let i = 0; i < n; i++) {
  jamHours[i] = jam[i] / perJamHour;
  etaHours[i] = eta[i] === NO_ETA ? -1 : eta[i] / sim.stepsPerHour;
  if (eta[i] !== NO_ETA) jammedCells++;
  dens[i] = density[i];
}
await Deno.writeFile(`${outDir}/jam.bin`, new Uint8Array(jamHours.buffer));
await Deno.writeFile(`${outDir}/eta.bin`, new Uint8Array(etaHours.buffer));
await Deno.writeFile(`${outDir}/density.bin`, new Uint8Array(dens.buffer));

const endParticles = await sim.particleStats();
const stats = {
  width: W,
  height: H,
  dx_m: wind.dx_m,
  dt_seconds: sim.dtSeconds,
  steps_per_hour: sim.stepsPerHour,
  spinup_hours: spinup,
  state_step: stateStep,
  seeded_particles: sim.seedParticles,
  forecast_hours: forecastHours,
  forecast_start: wind.forecast_start,
  barrage_open: wind.barrage_open,
  synthetic: !!wind.synthetic,
  particles_per_full_cell: sim.particlesPerSeedCell,
  jam_threshold_particles: sim.jamThreshold,
  particles_start: startParticles,
  particles_end: endParticles,
  jammed_cells: jammedCells,
  video: writer ? "forecast.mp4" : null,
  wall_seconds: (performance.now() - t0) / 1000,
};
await Deno.writeTextFile(`${outDir}/stats.json`, JSON.stringify(stats, null, 2));
console.log(
  `[render] done in ${stats.wall_seconds.toFixed(1)} s: ${jammedCells} jammed cells, ` +
    `${endParticles.stuck}/${endParticles.alive} mats stuck, ${endParticles.flushed} flushed`,
);
sim.destroy();
device.destroy();
