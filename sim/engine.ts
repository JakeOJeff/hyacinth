// Hyacinth simulation engine. Runtime-agnostic: the same class drives the
// browser demo (Vite + WebGPU) and the headless daily render (Deno WebGPU).
// The caller supplies the GPUDevice and the WGSL source strings.

import type { SimData, WindForcing } from "./data.ts";

export interface ShaderSources {
  common: string;
  lbm: string;
  particles: string;
  accumulate: string;
  render: string;
}

/** Physical knobs. Everything here is in physical units unless noted. */
export interface PhysParams {
  /** Seconds per simulation step. 0 = pick automatically from dx. */
  dtSeconds: number;
  /** BGK relaxation time. Must stay above ~0.55 or the solver blows up. */
  tau: number;
  /** Depth-averaged current produced per m/s of wind (m/s per m/s). */
  currentPerWind: number;
  /** Linear bottom friction, per step (lattice). Sets the spin-up time. */
  dragPerStep: number;
  /** Mat drift from windage, as a fraction of wind speed (~3 %). */
  windage: number;
  /** Random walk amplitude, metres per step. */
  jitterMeters: number;
  /** Probability per hour that a mat stuck on a bank breaks free. */
  releasePerHour: number;
  /** Particles representing a fully mat-covered cell in seed.png. */
  particlesPerSeedCell: number;
  /** Fraction of a fully covered cell that counts as "blocked". */
  jamFraction: number;
  /** A cell counts as jammed once blocked for this long (hours). */
  jamSustainHours: number;
  /** Jams are only counted within this distance of a bank or the barrage. */
  shoreBandMeters: number;
  /** Cohesion strength (cells per step at a strong density gradient). */
  cohesion: number;
  /** Lattice density excess at river inflow cells (pressure head). */
  riverHead: number;
  /** Lattice density deficit at sea outlet cells. */
  seaHead: number;
  /** Lattice velocity clamp (stability guard). */
  maxU: number;
  /** Upper bound on particle buffer size. */
  maxParticles: number;
  /** Extra slots for click-to-drop hyacinth. */
  dropSlots: number;
  rngSeed: number;
}

export const DEFAULT_PARAMS: PhysParams = {
  dtSeconds: 0,
  tau: 0.6,
  currentPerWind: 0.02,
  dragPerStep: 0.002,
  windage: 0.03,
  jitterMeters: 0.6,
  releasePerHour: 0.15,
  particlesPerSeedCell: 4,
  jamFraction: 0.75,
  jamSustainHours: 2,
  shoreBandMeters: 120,
  cohesion: 0.004,
  riverHead: 0.006,
  seaHead: 0.002,
  maxU: 0.12,
  maxParticles: 1_500_000,
  dropSlots: 200_000,
  rngSeed: 1337,
};

/** Highest physical speed (m/s) the automatic time step is sized for. */
const DESIGN_SPEED = 0.5;
/** Lattice speed that DESIGN_SPEED maps to. Well under maxU and Ma ~ 0.14. */
const DESIGN_LATTICE_SPEED = 0.08;

export interface WindOverride {
  speed: number; // m/s
  dirFrom: number; // meteorological degrees, direction the wind blows FROM
}

export interface JamReadback {
  /** Dwell time per cell in particle-steps since recordStart. */
  jam: Uint32Array;
  /** Step (relative to recordStart) at which the cell first jammed, or NO_ETA. */
  eta: Uint32Array;
  /** Current particle count per cell. */
  density: Uint32Array;
}

export const NO_ETA = 0xffffffff;
const PARAMS_BYTES = 112;
/** Per-step uniform stride in the batch buffer (minUniformBufferOffsetAlignment). */
const STEP_STRIDE = 256;
/** Steps encoded per submit; larger advances are split into several submits. */
const MAX_BATCH = 1024;
/** Must match TILE_ROW in common.wgsl. */
const TILE_ROW = 32768;

export interface AdvanceOptions {
  /** false = water only (mats frozen, jam/density untouched). Default true. */
  particles?: boolean;
}

/** GPU-side copy of the full sim state, restored without any CPU round trip. */
interface Checkpoint {
  f: [GPUBuffer, GPUBuffer];
  vel: GPUBuffer;
  particles: GPUBuffer;
  density: GPUBuffer;
  step: number;
  dropCursor: number;
  dropUsed: number;
}

export class HyacinthSim {
  readonly width: number;
  readonly height: number;
  readonly cells: number;
  readonly dtSeconds: number;
  readonly stepsPerHour: number;
  readonly particlesPerSeedCell: number;
  readonly jamThreshold: number;

  step = 0;
  recordStart = 0;
  barrageOpen: boolean;
  windOverride: WindOverride | null = null;
  viewMode: 0 | 1 = 0;
  showObserved = false;
  progress = 0;

  /** Particles actually placed from seed.png on the last reset. */
  seededCount = 0;
  private dropCursor = 0;
  private dropUsed = 0;
  private readonly batchBuf: GPUBuffer;
  private readonly batchData = new ArrayBuffer(MAX_BATCH * STEP_STRIDE);
  private checkpointState: Checkpoint | null = null;
  /** Number of 8x8 tiles containing water (the only ones the solver touches). */
  readonly activeTiles: number;
  private readonly tileDispatch: [number, number];
  private readonly seedParticleCount: number;
  private readonly capacity: number;
  private readonly params: PhysParams;
  private readonly paramsData = new ArrayBuffer(PARAMS_BYTES);

  private readonly paramsBuf: GPUBuffer;
  private readonly cellTypeBuf: GPUBuffer;
  private readonly fBufs: [GPUBuffer, GPUBuffer];
  private readonly velBuf: GPUBuffer;
  private readonly particleBuf: GPUBuffer;
  private readonly countBuf: GPUBuffer;
  private readonly densityBuf: GPUBuffer;
  private readonly jamBuf: GPUBuffer;
  private readonly etaBuf: GPUBuffer;
  private readonly runBuf: GPUBuffer;
  private readonly shoreBuf: GPUBuffer;
  private readonly observedBuf: GPUBuffer;
  private readonly cellInfoBuf: GPUBuffer;
  private readonly tileBuf: GPUBuffer;

  private readonly lbmPipe: GPUComputePipeline;
  private readonly partPipe: GPUComputePipeline;
  private readonly accPipe: GPUComputePipeline;
  private readonly lbmGroups: [GPUBindGroup, GPUBindGroup];
  private readonly partGroup: GPUBindGroup;
  private readonly accGroup: GPUBindGroup;
  private readonly renderModule: GPUShaderModule;
  private readonly renderPipes = new Map<GPUTextureFormat, { pipe: GPURenderPipeline; group: GPUBindGroup }>();

  private constructor(
    readonly device: GPUDevice,
    readonly data: SimData,
    shaders: ShaderSources,
    params: PhysParams,
  ) {
    this.params = params;
    this.width = data.width;
    this.height = data.height;
    this.cells = data.width * data.height;
    this.barrageOpen = data.wind.barrage_open;

    if (params.tau < 0.55) {
      throw new Error(`tau=${params.tau} is unstable; keep it >= 0.55`);
    }
    const dx = data.wind.dx_m;
    this.dtSeconds = params.dtSeconds > 0 ? params.dtSeconds : (DESIGN_LATTICE_SPEED * dx) / DESIGN_SPEED;
    this.stepsPerHour = Math.max(1, Math.round(3600 / this.dtSeconds));

    // Seed particles from seed.png density.
    const seedInfo = countSeedParticles(data, params);
    this.particlesPerSeedCell = seedInfo.perCell;
    this.jamThreshold = Math.max(1, params.jamFraction * seedInfo.perCell);
    this.seedParticleCount = seedInfo.total;
    this.capacity = this.seedParticleCount + params.dropSlots;

    const S = GPUBufferUsage.STORAGE;
    const CD = GPUBufferUsage.COPY_DST;
    const CS = GPUBufferUsage.COPY_SRC;
    const n = this.cells;

    this.paramsBuf = device.createBuffer({ size: PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | CD });
    this.cellTypeBuf = device.createBuffer({ size: n * 4, usage: S | CD });
    this.fBufs = [
      device.createBuffer({ size: n * 9 * 4, usage: S | CD | CS }),
      device.createBuffer({ size: n * 9 * 4, usage: S | CD | CS }),
    ];
    this.velBuf = device.createBuffer({ size: n * 8, usage: S | CD | CS });
    this.particleBuf = device.createBuffer({ size: this.capacity * 16, usage: S | CD | CS });
    this.countBuf = device.createBuffer({ size: n * 4, usage: S | CD });
    this.densityBuf = device.createBuffer({ size: n * 4, usage: S | CD | CS });
    this.jamBuf = device.createBuffer({ size: n * 4, usage: S | CD | CS });
    this.etaBuf = device.createBuffer({ size: n * 4, usage: S | CD | CS });
    this.runBuf = device.createBuffer({ size: n * 4, usage: S | CD });
    this.shoreBuf = device.createBuffer({ size: n * 4, usage: S | CD });
    this.observedBuf = device.createBuffer({ size: n * 4, usage: S | CD });
    this.batchBuf = device.createBuffer({ size: MAX_BATCH * STEP_STRIDE, usage: GPUBufferUsage.UNIFORM | CD });

    const tiles = waterTiles(data);
    this.activeTiles = tiles.length;
    this.tileDispatch = [Math.min(tiles.length, TILE_ROW), Math.ceil(tiles.length / TILE_ROW)];
    this.tileBuf = device.createBuffer({ size: Math.max(4, tiles.byteLength), usage: S | CD });
    device.queue.writeBuffer(this.tileBuf, 0, tiles);
    this.cellInfoBuf = device.createBuffer({ size: n * 4, usage: S | CD });
    device.queue.writeBuffer(this.cellInfoBuf, 0, cellInfo(data));

    device.queue.writeBuffer(this.cellTypeBuf, 0, data.cellType);
    device.queue.writeBuffer(this.shoreBuf, 0, shoreBand(data, Math.max(1, Math.round(params.shoreBandMeters / data.wind.dx_m))));
    const obs = new Uint32Array(n);
    if (data.observed) obs.set(data.observed);
    device.queue.writeBuffer(this.observedBuf, 0, obs);

    const mod = (src: string, label: string) => {
      const m = device.createShaderModule({ label, code: shaders.common + "\n" + src });
      return m;
    };
    const lbmMod = mod(shaders.lbm, "lbm");
    const partMod = mod(shaders.particles, "particles");
    const accMod = mod(shaders.accumulate, "accumulate");
    this.renderModule = mod(shaders.render, "render");

    // Explicit layouts: binding 0 is the per-step uniform slice of the batch
    // buffer, selected with a dynamic offset, so a whole batch of steps is one
    // compute pass with no buffer copies in between.
    const C = GPUShaderStage.COMPUTE;
    const uni: GPUBindGroupLayoutEntry = {
      binding: 0,
      visibility: C,
      buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: PARAMS_BYTES },
    };
    const ro = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: C, buffer: { type: "read-only-storage" } });
    const rw = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: C, buffer: { type: "storage" } });
    const lbmLayout = device.createBindGroupLayout({ label: "lbm", entries: [uni, ro(1), ro(2), rw(3), rw(4), ro(5), ro(6)] });
    const partLayout = device.createBindGroupLayout({ label: "particles", entries: [uni, ro(1), ro(2), ro(3), rw(4), rw(5)] });
    const accLayout = device.createBindGroupLayout({ label: "accumulate", entries: [uni, rw(1), rw(2), rw(3), rw(4), rw(5), ro(6), ro(7)] });
    const pipe = (label: string, module: GPUShaderModule, layout: GPUBindGroupLayout) =>
      device.createComputePipeline({
        label,
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module, entryPoint: "main" },
      });
    this.lbmPipe = pipe("lbm", lbmMod, lbmLayout);
    this.partPipe = pipe("particles", partMod, partLayout);
    this.accPipe = pipe("accumulate", accMod, accLayout);

    const stepSlice = { buffer: this.batchBuf, offset: 0, size: PARAMS_BYTES };
    const group = (layout: GPUBindGroupLayout, buffers: GPUBuffer[]) =>
      device.createBindGroup({
        layout,
        entries: [{ binding: 0, resource: stepSlice }, ...buffers.map((buffer, i) => ({ binding: i + 1, resource: { buffer } }))],
      });
    const lbmGroup = (fin: GPUBuffer, fout: GPUBuffer) =>
      group(lbmLayout, [this.cellTypeBuf, fin, fout, this.velBuf, this.cellInfoBuf, this.tileBuf]);
    this.lbmGroups = [lbmGroup(this.fBufs[0], this.fBufs[1]), lbmGroup(this.fBufs[1], this.fBufs[0])];
    this.partGroup = group(partLayout, [this.cellTypeBuf, this.velBuf, this.densityBuf, this.particleBuf, this.countBuf]);
    this.accGroup = group(accLayout, [this.countBuf, this.densityBuf, this.jamBuf, this.etaBuf, this.runBuf, this.shoreBuf, this.tileBuf]);

    this.reset();
  }

  /** Create the simulation. Throws with the WGSL compiler log on shader errors. */
  static async create(
    device: GPUDevice,
    data: SimData,
    shaders: ShaderSources,
    overrides: Partial<PhysParams> = {},
  ): Promise<HyacinthSim> {
    const params = { ...DEFAULT_PARAMS, ...overrides };
    device.pushErrorScope("validation");
    const sim = new HyacinthSim(device, data, shaders, params);
    sim.getRenderPipeline("rgba8unorm"); // compile the render shader up front
    const err = await device.popErrorScope();
    if (err) throw new Error(`WebGPU validation error while building the sim: ${err.message}`);
    return sim;
  }

  get particleCapacity(): number {
    return this.capacity;
  }

  get seedParticles(): number {
    return this.seedParticleCount;
  }

  /** Particle slots that may hold a live particle (seeded + drops used so far). */
  get activeParticleSlots(): number {
    return this.seedParticleCount + this.dropUsed;
  }

  /** Read-only access to the GPU buffers, for renderers that draw straight from sim state. */
  get buffers() {
    return {
      params: this.paramsBuf,
      cellType: this.cellTypeBuf,
      vel: this.velBuf,
      density: this.densityBuf,
      jam: this.jamBuf,
      particles: this.particleBuf,
    };
  }

  /** Hours since the start of the wind record (t0 = data.wind.start). */
  get simHours(): number {
    return this.step / this.stepsPerHour;
  }

  /** Knobs that can change while running (others are baked into buffers). */
  tune(
    overrides: Partial<Pick<PhysParams, "riverHead" | "seaHead" | "windage" | "currentPerWind" | "cohesion" | "releasePerHour">>,
  ): void {
    Object.assign(this.params, overrides);
  }

  get physParams(): Readonly<PhysParams> {
    return this.params;
  }

  /** Restore water to rest and re-seed mats from seed.png. Clears jam/ETA. */
  reset(): void {
    const n = this.cells;
    const q = this.device.queue;
    const W = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
    const f = new Float32Array(n * 9);
    for (let i = 0; i < 9; i++) f.fill(W[i], i * n, (i + 1) * n);
    q.writeBuffer(this.fBufs[0], 0, f);
    q.writeBuffer(this.fBufs[1], 0, f);
    q.writeBuffer(this.velBuf, 0, new Float32Array(n * 2));
    q.writeBuffer(this.countBuf, 0, new Uint32Array(n));
    q.writeBuffer(this.densityBuf, 0, new Uint32Array(n));
    this.clearJam();

    const parts = new Float32Array(this.capacity * 4);
    this.seededCount = seedParticles(this.data, this.particlesPerSeedCell, this.params.rngSeed, parts);
    q.writeBuffer(this.particleBuf, 0, parts);
    this.step = 0;
    this.dropCursor = 0;
    this.dropUsed = 0;
  }

  /** Zero the jam / ETA accumulators and start recording at the current step. */
  clearJam(): void {
    const n = this.cells;
    this.device.queue.writeBuffer(this.jamBuf, 0, new Uint32Array(n));
    this.device.queue.writeBuffer(this.etaBuf, 0, new Uint32Array(n).fill(NO_ETA));
    this.device.queue.writeBuffer(this.runBuf, 0, new Uint32Array(n));
    this.recordStart = this.step;
  }

  /** Wind (east, north) in m/s at the current sim time. */
  currentWind(): { u: number; v: number; speed: number; dirFrom: number } {
    if (this.windOverride) {
      const { speed, dirFrom } = this.windOverride;
      const r = (dirFrom * Math.PI) / 180;
      return { u: -speed * Math.sin(r), v: -speed * Math.cos(r), speed, dirFrom };
    }
    const { u, v } = windAt(this.data.wind, this.simHours);
    const speed = Math.hypot(u, v);
    const dirFrom = (((Math.atan2(-u, -v) * 180) / Math.PI) + 360) % 360;
    return { u, v, speed, dirFrom };
  }

  /** Upload the per-step uniforms (also used by external renderers). */
  writeParams(): void {
    this.packParams(new DataView(this.paramsData), 0);
    this.device.queue.writeBuffer(this.paramsBuf, 0, this.paramsData);
  }

  /** Pack the uniforms for the current step into dv at byte offset base. */
  private packParams(dv: DataView, base: number): void {
    const p = this.params;
    const dx = this.data.wind.dx_m;
    const k = this.dtSeconds / dx; // lattice velocity per m/s
    const w = this.currentWind();
    // Lattice y grows south, so the north component flips sign.
    const wx = w.u;
    const wy = -w.v;
    const cw = p.dragPerStep * p.currentPerWind * k; // steady state: u = cw*W/drag
    const drift = p.windage * k;
    const release = 1 - Math.pow(1 - Math.min(0.999, p.releasePerHour), 1 / this.stepsPerHour);

    const u32 = (o: number, v: number) => dv.setUint32(base + o, v >>> 0, true);
    const f32 = (o: number, v: number) => dv.setFloat32(base + o, v, true);
    u32(0, this.width);
    u32(4, this.height);
    u32(8, this.step);
    u32(12, this.activeParticleSlots);
    f32(16, p.tau);
    f32(20, 1 + p.riverHead);
    f32(24, 1 - p.seaHead);
    u32(28, this.barrageOpen ? 1 : 0);
    f32(32, cw * wx);
    f32(36, cw * wy);
    f32(40, p.dragPerStep);
    f32(44, p.maxU);
    f32(48, drift * wx);
    f32(52, drift * wy);
    f32(56, p.jitterMeters / dx);
    f32(60, release);
    f32(64, this.jamThreshold);
    f32(68, this.particlesPerSeedCell);
    f32(72, p.cohesion);
    u32(76, p.rngSeed);
    u32(80, this.viewMode);
    u32(84, this.showObserved && this.data.observed ? 1 : 0);
    f32(88, this.jamThreshold * this.stepsPerHour); // jamUnit: one jam-hour
    f32(92, this.progress);
    u32(96, this.recordStart);
    u32(100, Math.max(1, Math.round(p.jamSustainHours * this.stepsPerHour)));
  }

  /**
   * Advance n steps. Every step's uniforms (step counter, wind) are packed up
   * front into a 256-byte-strided uniform buffer and picked per dispatch with
   * a dynamic offset, so a batch of up to MAX_BATCH steps is one compute pass
   * and one submit. Results match stepping one at a time.
   */
  advance(n: number, opts: AdvanceOptions = {}): void {
    const withParticles = opts.particles !== false;
    const [tx, ty] = this.tileDispatch;
    while (n > 0) {
      const m = Math.min(n, MAX_BATCH);
      n -= m;
      const dv = new DataView(this.batchData);
      const start = this.step;
      for (let s = 0; s < m; s++) {
        this.step = start + s;
        this.packParams(dv, s * STEP_STRIDE);
      }
      this.step = start;
      this.device.queue.writeBuffer(this.batchBuf, 0, this.batchData, 0, m * STEP_STRIDE);
      // Particles only exist in seeded + used drop slots; dispatch just those.
      const gp = Math.ceil(this.activeParticleSlots / 256);
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (let s = 0; s < m; s++) {
        const off = [s * STEP_STRIDE];
        pass.setPipeline(this.lbmPipe);
        pass.setBindGroup(0, this.lbmGroups[this.step & 1], off);
        pass.dispatchWorkgroups(tx, ty);
        if (withParticles) {
          if (gp > 0) {
            pass.setPipeline(this.partPipe);
            pass.setBindGroup(0, this.partGroup, off);
            pass.dispatchWorkgroups(gp);
          }
          pass.setPipeline(this.accPipe);
          pass.setBindGroup(0, this.accGroup, off);
          pass.dispatchWorkgroups(tx, ty);
        }
        this.step++;
      }
      pass.end();
      this.device.queue.submit([enc.finish()]);
    }
  }

  /** Snapshot the whole state on the GPU (water, mats, step). Buffer copies only. */
  checkpoint(): void {
    const d = this.device;
    const usage = GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const ck: Checkpoint = this.checkpointState ?? {
      f: [d.createBuffer({ size: this.fBufs[0].size, usage }), d.createBuffer({ size: this.fBufs[1].size, usage })],
      vel: d.createBuffer({ size: this.velBuf.size, usage }),
      particles: d.createBuffer({ size: this.particleBuf.size, usage }),
      density: d.createBuffer({ size: this.densityBuf.size, usage }),
      step: 0,
      dropCursor: 0,
      dropUsed: 0,
    };
    this.copyState(
      [this.fBufs[0], this.fBufs[1], this.velBuf, this.particleBuf, this.densityBuf],
      [ck.f[0], ck.f[1], ck.vel, ck.particles, ck.density],
    );
    ck.step = this.step;
    ck.dropCursor = this.dropCursor;
    ck.dropUsed = this.dropUsed;
    this.checkpointState = ck;
  }

  get checkpointStep(): number | null {
    return this.checkpointState?.step ?? null;
  }

  /** Return to the last checkpoint and start recording jams from there. */
  restore(): boolean {
    const ck = this.checkpointState;
    if (!ck) return false;
    this.copyState(
      [ck.f[0], ck.f[1], ck.vel, ck.particles, ck.density],
      [this.fBufs[0], this.fBufs[1], this.velBuf, this.particleBuf, this.densityBuf],
    );
    this.device.queue.writeBuffer(this.countBuf, 0, new Uint32Array(this.cells));
    this.step = ck.step;
    this.dropCursor = ck.dropCursor;
    this.dropUsed = ck.dropUsed;
    this.clearJam();
    return true;
  }

  private copyState(from: GPUBuffer[], to: GPUBuffer[]) {
    const enc = this.device.createCommandEncoder();
    from.forEach((src, i) => enc.copyBufferToBuffer(src, 0, to[i], 0, src.size));
    this.device.queue.submit([enc.finish()]);
  }

  /**
   * Replace the seeded mats with a saved snapshot (vec4 per seeded slot, as
   * written by the daily render at forecast start) and jump to `step`.
   * Water is left as it is: run a short particle-free warm-up afterwards.
   */
  loadParticles(state: Float32Array, step: number): boolean {
    if (state.length !== this.seedParticleCount * 4) return false;
    const parts = new Float32Array(this.capacity * 4);
    parts.set(state);
    this.device.queue.writeBuffer(this.particleBuf, 0, parts);
    this.device.queue.writeBuffer(this.countBuf, 0, new Uint32Array(this.cells));
    this.device.queue.writeBuffer(this.densityBuf, 0, new Uint32Array(this.cells));
    this.step = step;
    this.dropCursor = 0;
    this.dropUsed = 0;
    this.clearJam();
    return true;
  }

  /** Seeded particle slots only (for the forecast-start snapshot). */
  async readSeededParticles(): Promise<Float32Array> {
    return (await this.readParticles()).slice(0, this.seedParticleCount * 4);
  }

  private getRenderPipeline(format: GPUTextureFormat) {
    let entry = this.renderPipes.get(format);
    if (!entry) {
      const pipe = this.device.createRenderPipeline({
        label: `render-${format}`,
        layout: "auto",
        vertex: { module: this.renderModule, entryPoint: "vs" },
        fragment: { module: this.renderModule, entryPoint: "fs", targets: [{ format }] },
        primitive: { topology: "triangle-list" },
      });
      const group = this.device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.paramsBuf } },
          { binding: 1, resource: { buffer: this.cellTypeBuf } },
          { binding: 2, resource: { buffer: this.velBuf } },
          { binding: 3, resource: { buffer: this.densityBuf } },
          { binding: 4, resource: { buffer: this.jamBuf } },
          { binding: 5, resource: { buffer: this.observedBuf } },
        ],
      });
      entry = { pipe, group };
      this.renderPipes.set(format, entry);
    }
    return entry;
  }

  /** Draw the current state into a width x height texture view. */
  render(view: GPUTextureView, format: GPUTextureFormat, encoder?: GPUCommandEncoder): void {
    this.writeParams();
    const { pipe, group } = this.getRenderPipeline(format);
    const enc = encoder ?? this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    pass.setPipeline(pipe);
    pass.setBindGroup(0, group);
    pass.draw(3);
    pass.end();
    if (!encoder) this.device.queue.submit([enc.finish()]);
  }

  /** Drop a blob of hyacinth at cell (cx, cy). Returns particles added. */
  dropHyacinth(cx: number, cy: number, radiusCells: number, count: number): number {
    const slots = this.params.dropSlots;
    if (slots <= 0) return 0;
    count = Math.min(count, slots);
    const out = new Float32Array(count * 4);
    let added = 0;
    let tries = 0;
    let rng = (this.params.rngSeed ^ (this.step * 2654435761)) >>> 0;
    const rand = () => {
      rng = (rng + 0x6d2b79f5) >>> 0;
      let t = rng;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    while (added < count && tries < count * 20) {
      tries++;
      const a = rand() * Math.PI * 2;
      const r = Math.sqrt(rand()) * radiusCells;
      const x = cx + Math.cos(a) * r;
      const y = cy + Math.sin(a) * r;
      const ix = Math.floor(x);
      const iy = Math.floor(y);
      if (ix < 0 || iy < 0 || ix >= this.width || iy >= this.height) continue;
      const t = this.data.cellType[iy * this.width + ix];
      if (t !== 1 && t !== 2) continue;
      out.set([x, y, 1, 0], added * 4);
      added++;
    }
    // Write into the ring of drop slots after the seeded particles.
    let written = 0;
    while (written < added) {
      const n = Math.min(added - written, slots - this.dropCursor);
      const offset = (this.seedParticleCount + this.dropCursor) * 16;
      this.device.queue.writeBuffer(this.particleBuf, offset, out, written * 4, n * 4);
      written += n;
      this.dropCursor = (this.dropCursor + n) % slots;
      this.dropUsed = Math.min(slots, this.dropUsed + n);
    }
    return added;
  }

  /** Copy the jam, ETA and density buffers back to the CPU. */
  async readJam(): Promise<JamReadback> {
    const n = this.cells * 4;
    const staging = this.device.createBuffer({ size: n * 3, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.jamBuf, 0, staging, 0, n);
    enc.copyBufferToBuffer(this.etaBuf, 0, staging, n, n);
    enc.copyBufferToBuffer(this.densityBuf, 0, staging, 2 * n, n);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const all = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return {
      jam: all.slice(0, this.cells),
      eta: all.slice(this.cells, 2 * this.cells),
      density: all.slice(2 * this.cells),
    };
  }

  /** Velocity field (lattice units, cells/step) and total fluid mass. For tests. */
  async readFlow(): Promise<{ vel: Float32Array; mass: number }> {
    const n = this.cells;
    const fBytes = n * 9 * 4;
    const staging = this.device.createBuffer({ size: fBytes + n * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    // After an odd number of steps the newest distributions are in fBufs[1].
    enc.copyBufferToBuffer(this.fBufs[this.step & 1], 0, staging, 0, fBytes);
    enc.copyBufferToBuffer(this.velBuf, 0, staging, fBytes, n * 8);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const range = staging.getMappedRange();
    const f = new Float32Array(range, 0, n * 9);
    let mass = 0;
    for (let c = 0; c < n; c++) {
      const t = this.data.cellType[c];
      if (t === 1) for (let i = 0; i < 9; i++) mass += f[i * n + c];
    }
    const vel = new Float32Array(range.slice(fBytes));
    staging.unmap();
    staging.destroy();
    return { vel, mass };
  }

  /** Raw particle buffer: vec4(x, y, state, age) per slot. For tests. */
  async readParticles(): Promise<Float32Array> {
    const size = this.capacity * 16;
    const staging = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.particleBuf, 0, staging, 0, size);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  /** Count live / stuck particles (CPU readback, for stats). */
  async particleStats(): Promise<{ alive: number; stuck: number; flushed: number }> {
    const size = this.capacity * 16;
    const staging = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.particleBuf, 0, staging, 0, size);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const p = new Float32Array(staging.getMappedRange());
    let alive = 0;
    let stuck = 0;
    let flushed = 0;
    for (let i = 0; i < this.capacity; i++) {
      const s = p[i * 4 + 2];
      if (s > 1.5) stuck++;
      if (s > 0.5) alive++;
      else if (p[i * 4 + 3] > 0) flushed++;
    }
    staging.unmap();
    staging.destroy();
    return { alive, stuck, flushed };
  }

  destroy(): void {
    for (const b of [
      this.paramsBuf, this.cellTypeBuf, this.fBufs[0], this.fBufs[1], this.velBuf,
      this.particleBuf, this.countBuf, this.densityBuf, this.jamBuf, this.etaBuf, this.runBuf, this.shoreBuf, this.observedBuf,
      this.batchBuf, this.cellInfoBuf, this.tileBuf,
      ...(this.checkpointState
        ? [...this.checkpointState.f, this.checkpointState.vel, this.checkpointState.particles, this.checkpointState.density]
        : []),
    ]) b.destroy();
    this.checkpointState = null;
  }
}

/** Linear interpolation of the hourly wind record; holds the ends. */
export function windAt(w: WindForcing, hours: number): { u: number; v: number } {
  const n = w.u.length;
  if (n === 0) return { u: 0, v: 0 };
  const t = hours / w.step_hours;
  if (t <= 0) return { u: w.u[0], v: w.v[0] };
  if (t >= n - 1) return { u: w.u[n - 1], v: w.v[n - 1] };
  const i = Math.floor(t);
  const f = t - i;
  return { u: w.u[i] * (1 - f) + w.u[i + 1] * f, v: w.v[i] * (1 - f) + w.v[i + 1] * f };
}

function isWaterType(t: number): boolean {
  return t === 1 || t === 2;
}

function countSeedParticles(data: SimData, p: PhysParams): { perCell: number; total: number } {
  let coverage = 0;
  for (let c = 0; c < data.width * data.height; c++) {
    if (isWaterType(data.cellType[c])) coverage += data.seed[c] / 255;
  }
  let perCell = p.particlesPerSeedCell;
  const budget = p.maxParticles - p.dropSlots;
  if (coverage * perCell > budget) perCell = budget / coverage;
  // Fractional particles are drawn stochastically; +1 per cell is a safe bound.
  let total = 0;
  for (let c = 0; c < data.width * data.height; c++) {
    if (isWaterType(data.cellType[c]) && data.seed[c] > 0) total += Math.ceil((data.seed[c] / 255) * perCell);
  }
  return { perCell, total };
}

function seedParticles(data: SimData, perCell: number, seed: number, out: Float32Array): number {
  let s = seed >>> 0 || 1;
  const rand = () => {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 4294967296;
  };
  let k = 0;
  const cap = out.length / 4;
  for (let y = 0; y < data.height; y++) {
    for (let x = 0; x < data.width; x++) {
      const c = y * data.width + x;
      if (!isWaterType(data.cellType[c]) || data.seed[c] === 0) continue;
      const expect = (data.seed[c] / 255) * perCell;
      let m = Math.floor(expect);
      if (rand() < expect - m) m++;
      for (let j = 0; j < m && k < cap; j++) {
        out[k * 4] = x + rand();
        out[k * 4 + 1] = y + rand();
        out[k * 4 + 2] = 1;
        out[k * 4 + 3] = 0;
        k++;
      }
    }
  }
  return k;
}

/**
 * 1 for water cells within `band` cells (Chebyshev distance) of land or the
 * barrage: where piled-up mats actually block a channel.
 */
export function shoreBand(data: SimData, band: number): Uint32Array<ArrayBuffer> {
  const { width: w, height: h, cellType } = data;
  const n = w * h;
  const dist = new Int32Array(n).fill(-1);
  let frontier: number[] = [];
  for (let c = 0; c < n; c++) {
    if (cellType[c] === 0 || cellType[c] === 4) {
      dist[c] = 0;
      frontier.push(c);
    }
  }
  for (let d = 1; d <= band && frontier.length; d++) {
    const next: number[] = [];
    for (const c of frontier) {
      const x = c % w;
      const y = (c - x) / w;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const k = ny * w + nx;
          if (dist[k] !== -1) continue;
          dist[k] = d;
          next.push(k);
        }
      }
    }
    frontier = next;
  }
  const out = new Uint32Array(n);
  for (let c = 0; c < n; c++) {
    const t = cellType[c];
    if ((t === 1 || t === 2) && dist[c] > 0) out[c] = 1;
  }
  return out;
}

/** 8x8 tiles that contain any non-land cell, packed as tx | ty << 16. */
export function waterTiles(data: SimData): Uint32Array<ArrayBuffer> {
  const { width: w, height: h, cellType } = data;
  const tw = Math.ceil(w / 8);
  const th = Math.ceil(h / 8);
  const out: number[] = [];
  for (let ty = 0; ty < th; ty++) {
    for (let tx = 0; tx < tw; tx++) {
      let any = false;
      for (let y = ty * 8; y < Math.min(h, ty * 8 + 8) && !any; y++) {
        for (let x = tx * 8; x < Math.min(w, tx * 8 + 8); x++) {
          if (cellType[y * w + x] !== 0) {
            any = true;
            break;
          }
        }
      }
      if (any) out.push(tx | (ty << 16));
    }
  }
  return new Uint32Array(out);
}

const LATTICE_CX = [0, 1, 0, -1, 0, 1, -1, -1, 1];
const LATTICE_CY = [0, 0, 1, 0, -1, 1, 1, -1, -1];

/**
 * Per cell: bits 0-3 type, bits 4-12 "streaming source for direction i is land
 * or off-grid", bits 13-21 "source is the barrage" (a wall only while closed).
 */
export function cellInfo(data: SimData): Uint32Array<ArrayBuffer> {
  const { width: w, height: h, cellType } = data;
  const out = new Uint32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const c = y * w + x;
      let v = cellType[c] & 15;
      for (let i = 0; i < 9; i++) {
        const sx = x - LATTICE_CX[i];
        const sy = y - LATTICE_CY[i];
        if (sx < 0 || sy < 0 || sx >= w || sy >= h) {
          v |= 1 << (4 + i);
          continue;
        }
        const t = cellType[sy * w + sx];
        if (t === 0) v |= 1 << (4 + i);
        else if (t === 4) v |= 1 << (13 + i);
      }
      out[c] = v >>> 0;
    }
  }
  return out;
}
