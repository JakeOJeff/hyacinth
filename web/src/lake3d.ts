// Cinematic 3D view of the live simulation. Draws straight from the sim's GPU
// buffers (current, mats, jams) on the same device: no per-frame readback.

import type { HyacinthSim } from "../../sim/engine.ts";

type Vec3 = [number, number, number];

const CAM_BYTES = 208;

// ---------- small matrix helpers (column-major, WebGPU clip z in 0..1) ----------

function perspective(fovY: number, aspect: number, near: number, far: number): Float32Array<ArrayBuffer> {
  const f = 1 / Math.tan(fovY / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = far / (near - far);
  m[11] = -1;
  m[14] = (near * far) / (near - far);
  return m;
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: Vec3): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

function lookAt(eye: Vec3, target: Vec3, up: Vec3): Float32Array<ArrayBuffer> {
  const z = norm(sub(eye, target));
  const x = norm(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
}

function mul(a: Float32Array, b: Float32Array): Float32Array<ArrayBuffer> {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
  }
  return o;
}

function invert(m: Float32Array): Float32Array<ArrayBuffer> {
  const [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33] = m;
  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;
  const det = 1 / (b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06);
  return new Float32Array([
    (a11 * b11 - a12 * b10 + a13 * b09) * det,
    (a02 * b10 - a01 * b11 - a03 * b09) * det,
    (a31 * b05 - a32 * b04 + a33 * b03) * det,
    (a22 * b04 - a21 * b05 - a23 * b03) * det,
    (a12 * b08 - a10 * b11 - a13 * b07) * det,
    (a00 * b11 - a02 * b08 + a03 * b07) * det,
    (a32 * b02 - a30 * b05 - a33 * b01) * det,
    (a20 * b05 - a22 * b02 + a23 * b01) * det,
    (a10 * b10 - a11 * b08 + a13 * b06) * det,
    (a01 * b08 - a00 * b10 - a03 * b06) * det,
    (a30 * b04 - a31 * b02 + a33 * b00) * det,
    (a21 * b02 - a20 * b04 - a23 * b00) * det,
    (a11 * b07 - a10 * b09 - a12 * b06) * det,
    (a00 * b09 - a01 * b07 + a02 * b06) * det,
    (a31 * b01 - a30 * b03 - a32 * b00) * det,
    (a20 * b03 - a21 * b01 + a22 * b00) * det,
  ]);
}

function transform(m: Float32Array, v: [number, number, number, number]): Vec3 {
  const o = [0, 0, 0, 0];
  for (let r = 0; r < 4; r++) o[r] = m[r] * v[0] + m[4 + r] * v[1] + m[8 + r] * v[2] + m[12 + r] * v[3];
  return [o[0] / o[3], o[1] / o[3], o[2] / o[3]];
}

const smooth = (a: number, b: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

// ---------- static scene data ----------

/** Chessboard distance (capped) from every cell to the nearest source cell. */
function distanceFrom(isSource: (t: number) => boolean, ct: Uint32Array | Uint8Array, w: number, h: number, cap: number): Uint8Array {
  const n = w * h;
  const dist = new Uint8Array(n).fill(255);
  let frontier: number[] = [];
  for (let c = 0; c < n; c++) {
    if (isSource(ct[c])) {
      dist[c] = 0;
      frontier.push(c);
    }
  }
  for (let d = 1; d <= cap && frontier.length; d++) {
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
          if (dist[k] !== 255) continue;
          dist[k] = d;
          next.push(k);
        }
      }
    }
    frontier = next;
  }
  return dist;
}

function hash2(x: number, y: number): number {
  let h = (Math.imul(x | 0, 374761393) + Math.imul(y | 0, 668265263)) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function valueNoise(x: number, y: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const a = hash2(ix, iy);
  const b = hash2(ix + 1, iy);
  const c = hash2(ix, iy + 1);
  const d = hash2(ix + 1, iy + 1);
  return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
}

/**
 * Heights in cell units. Water deepens away from the banks; land sits low
 * (Kuttanad's polders are at or below sea level) behind raised embankments
 * along every waterway.
 */
function buildHeights(ct: Uint32Array | Uint8Array, w: number, h: number) {
  const isWater = (t: number) => t === 1 || t === 2 || t === 3;
  const toWater = distanceFrom(isWater, ct, w, h, 12);
  const toLand = distanceFrom((t) => t === 0, ct, w, h, 12);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const c = y * w + x;
      const t = ct[c];
      if (t === 4) {
        out[c] = 0;
      } else if (isWater(t)) {
        out[c] = -0.18 - 0.75 * smooth(1, 10, toLand[c]);
      } else {
        const n = valueNoise(x * 0.11, y * 0.11);
        out[c] = toWater[c] <= 1 ? 0.33 + 0.05 * n : 0.13 + 0.1 * smooth(2, 12, toWater[c]) + 0.06 * n;
      }
    }
  }
  for (let pass = 0; pass < 2; pass++) {
    const src = out.slice();
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const c = y * w + x;
        if (ct[c] === 4) continue;
        let s = 0;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) s += src[c + dy * w + dx];
        out[c] = s / 9;
      }
    }
  }
  // Keep every waterway open and every bank dry after smoothing.
  for (let c = 0; c < w * h; c++) {
    const t = ct[c];
    if (t === 1 || t === 2 || t === 3) out[c] = Math.min(out[c], -0.1);
    else if (t === 0) out[c] = Math.max(out[c], 0.06);
  }
  return { heights: out, toWater };
}

/** Coconut palms: dense along the canal banks, sparse inland. */
function placePalms(toWater: Uint8Array, ct: Uint32Array | Uint8Array, w: number, h: number, max: number): Float32Array<ArrayBuffer> {
  const prob = [0, 0.5, 0.24, 0.09, 0.04];
  const out: number[] = [];
  let s = 0x2f6b1d3 >>> 0;
  const rand = () => {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 4294967296;
  };
  for (let c = 0; c < w * h && out.length < max * 4; c++) {
    if (ct[c] !== 0) continue;
    const d = toWater[c];
    const p = d < prob.length ? prob[d] : 0.012;
    if (rand() >= p) continue;
    const x = c % w;
    const y = (c - x) / w;
    out.push(x + 0.15 + rand() * 0.7, y + 0.15 + rand() * 0.7, 0.75 + rand() * 0.55, rand());
  }
  return new Float32Array(out);
}

// ---------- the view ----------

export interface Lake3DFrame {
  /** Hour of day (local, 0..24) driving the sun. */
  hour: number;
  /** Wind toward east / north in m/s. */
  wind: { u: number; v: number; speed: number };
}

export class Lake3D {
  heatOn = true;
  autoOrbit = true;
  onPick: ((x: number, y: number) => void) | null = null;

  private readonly ctx: GPUCanvasContext | null;
  private readonly format: GPUTextureFormat;
  private readonly camBuf: GPUBuffer;
  private readonly camData = new Float32Array(CAM_BYTES / 4);
  private readonly group: GPUBindGroup;
  private readonly pipes: Record<"sky" | "terrain" | "skirt" | "water" | "mat" | "palm", GPURenderPipeline>;
  private readonly palmCount: number;
  private readonly terrainIndex: GPUBuffer;
  private readonly terrainIndexCount: number;
  private readonly stride: number;
  private readonly gridW: number;
  /** Fraction of device pixels rendered; adapts to keep frames smooth. */
  private scale = 1;
  private maxScale = 1;
  private frameMs = 16;
  private lastScaleChange = 0;
  private msaa: GPUTexture | null = null;
  private depth: GPUTexture | null = null;

  // Orbit camera around (tx, 0, tz).
  private yaw = 0.35;
  private pitch = 0.62;
  private dist = 900;
  private tx: number;
  private tz: number;
  private invVP = new Float32Array(16);
  private vp = new Float32Array(16);
  private fly: { t0: number; dur: number; from: [number, number, number]; to: [number, number, number] } | null = null;
  private lastInput = -1e9;
  private intro = { t0: performance.now(), dur: 4200 };
  private lastFrame = performance.now();

  private constructor(
    private readonly sim: HyacinthSim,
    private readonly canvas: HTMLCanvasElement | null,
    pipes: Lake3D["pipes"],
    group: GPUBindGroup,
    camBuf: GPUBuffer,
    palmCount: number,
    ctx: GPUCanvasContext | null,
    format: GPUTextureFormat,
    terrain: { index: GPUBuffer; count: number; stride: number; gridW: number },
  ) {
    this.terrainIndex = terrain.index;
    this.terrainIndexCount = terrain.count;
    this.stride = terrain.stride;
    this.gridW = terrain.gridW;
    this.maxScale = Math.min(typeof devicePixelRatio === "number" ? devicePixelRatio : 1, 1.5);
    this.scale = this.maxScale;
    this.pipes = pipes;
    this.group = group;
    this.camBuf = camBuf;
    this.palmCount = palmCount;
    this.ctx = ctx;
    this.format = format;
    this.tx = sim.width / 2;
    this.tz = sim.height / 2;
    this.resetView(false);
    if (canvas) this.bindControls(canvas);
  }

  /** Pass canvas = null to render offscreen with renderTo() (tests, headless stills). */
  static async create(
    sim: HyacinthSim,
    canvas: HTMLCanvasElement | null,
    common: string,
    source: string,
    format: GPUTextureFormat = navigator.gpu.getPreferredCanvasFormat(),
  ): Promise<Lake3D> {
    const device = sim.device;
    let ctx: GPUCanvasContext | null = null;
    if (canvas) {
      ctx = canvas.getContext("webgpu");
      if (!ctx) throw new Error("no WebGPU canvas context for the 3D view");
      ctx.configure({ device, format, alphaMode: "opaque" });
    }

    const { heights, toWater } = buildHeights(sim.data.cellType, sim.width, sim.height);
    const palms = placePalms(toWater, sim.data.cellType, sim.width, sim.height, 30_000);
    const palmCount = palms.length / 4;

    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const heightBuf = device.createBuffer({ size: heights.byteLength, usage: S });
    device.queue.writeBuffer(heightBuf, 0, heights);
    const palmBuf = device.createBuffer({ size: Math.max(16, palms.byteLength), usage: S });
    if (palmCount) device.queue.writeBuffer(palmBuf, 0, palms);
    const camBuf = device.createBuffer({ size: CAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // Terrain as an indexed grid (vertices shared between quads) at a coarser
    // stride on big maps: ~4x fewer vertices, ~6x fewer vertex shader runs.
    const stride = sim.width * sim.height > 300_000 ? 2 : 1;
    const gridW = Math.floor((sim.width - 1) / stride) + 1;
    const gridH = Math.floor((sim.height - 1) / stride) + 1;
    const idx = new Uint32Array((gridW - 1) * (gridH - 1) * 6);
    let k = 0;
    for (let j = 0; j < gridH - 1; j++) {
      for (let i = 0; i < gridW - 1; i++) {
        const a = j * gridW + i;
        idx[k++] = a;
        idx[k++] = a + 1;
        idx[k++] = a + gridW;
        idx[k++] = a + 1;
        idx[k++] = a + gridW + 1;
        idx[k++] = a + gridW;
      }
    }
    const index = device.createBuffer({ size: idx.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(index, 0, idx);

    const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    const ro = { type: "read-only-storage" as const };
    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: VF, buffer: { type: "uniform" } },
        { binding: 1, visibility: VF, buffer: { type: "uniform" } },
        ...[2, 3, 4, 5, 6, 7, 8].map((binding) => ({ binding, visibility: VF, buffer: ro })),
      ],
    });
    const b = sim.buffers;
    const group = device.createBindGroup({
      layout,
      entries: [b.params, camBuf, b.cellType, b.vel, b.density, b.jam, b.particles, heightBuf, palmBuf].map((buffer, binding) => ({
        binding,
        resource: { buffer },
      })),
    });

    const module = device.createShaderModule({ label: "lake3d", code: common + "\n" + source });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === "error");
    if (errors.length) throw new Error(`3D shader: ${errors.map((e) => `${e.lineNum}:${e.linePos} ${e.message}`).join("; ")}`);

    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const multisample = { count: 4 };
    const opaqueDepth: GPUDepthStencilState = { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" };
    const make = (vs: string, fs: string, opts: { depth: GPUDepthStencilState; blend?: GPUBlendState }) =>
      device.createRenderPipelineAsync({
        label: vs,
        layout: pipelineLayout,
        vertex: { module, entryPoint: vs },
        fragment: { module, entryPoint: fs, targets: [{ format, blend: opts.blend }] },
        primitive: { topology: "triangle-list", cullMode: "none" },
        depthStencil: opts.depth,
        multisample,
      });
    const [sky, terrain, skirt, water, mat, palm] = await Promise.all([
      // Drawn after the opaque geometry at depth 1: only uncovered pixels run the sky shader.
      make("skyVs", "skyFs", { depth: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "less-equal" } }),
      make("terrainVs", "terrainFs", { depth: opaqueDepth }),
      make("skirtVs", "terrainFs", { depth: opaqueDepth }),
      make("waterVs", "waterFs", {
        depth: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "less" },
        blend: {
          color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
        },
      }),
      make("matVs", "plantFs", { depth: opaqueDepth }),
      make("palmVs", "plantFs", { depth: opaqueDepth }),
    ]);
    return new Lake3D(sim, canvas, { sky, terrain, skirt, water, mat, palm }, group, camBuf, palmCount, ctx, format, { index, count: idx.length, stride, gridW });
  }

  /** Jump the camera (no fly-in). Angles in radians, distances in cells. */
  look(v: { yaw?: number; pitch?: number; dist?: number; x?: number; z?: number }): void {
    if (v.yaw !== undefined) this.yaw = v.yaw;
    if (v.pitch !== undefined) this.pitch = v.pitch;
    if (v.dist !== undefined) this.dist = v.dist;
    if (v.x !== undefined) this.tx = v.x;
    if (v.z !== undefined) this.tz = v.z;
    this.intro.dur = 0;
    this.lastInput = performance.now();
  }

  /** Smoothly move the camera to look at grid cell (x, z) from `dist` cells away. */
  flyTo(x: number, z: number, dist = 150): void {
    this.fly = { t0: performance.now(), dur: 1400, from: [this.tx, this.tz, this.dist], to: [x, z, dist] };
    this.intro.dur = 0;
    this.lastInput = performance.now();
  }

  /** Client (CSS pixel) coordinates of grid point (x, z) on the water, or null if behind the camera. */
  project(x: number, z: number): [number, number] | null {
    if (!this.canvas) return null;
    const m = this.vp;
    const cx = m[0] * x + m[8] * z + m[12];
    const cy = m[1] * x + m[9] * z + m[13];
    const cw = m[3] * x + m[11] * z + m[15];
    if (cw <= 0) return null;
    const r = this.canvas.getBoundingClientRect();
    return [r.left + ((cx / cw + 1) / 2) * r.width, r.top + ((1 - cy / cw) / 2) * r.height];
  }

  /** Fly back to the overview. */
  resetView(animate = true): void {
    this.tx = this.sim.width / 2;
    this.tz = this.sim.height * 0.55;
    this.yaw = 0.35;
    this.pitch = 0.62;
    this.dist = Math.max(this.sim.width, this.sim.height) * 0.78;
    this.intro = { t0: animate ? performance.now() : this.intro.t0, dur: animate ? 1600 : this.intro.dur };
    this.lastInput = -1e9;
  }

  private markInput() {
    this.fly = null;
    this.lastInput = performance.now();
    this.intro.dur = 0; // any input ends the fly-in
  }

  private bindControls(el: HTMLCanvasElement) {
    const pts = new Map<number, { x: number; y: number }>();
    let moved = 0;
    let pinch = 0;
    el.addEventListener("contextmenu", (e) => e.preventDefault());
    el.addEventListener("pointerdown", (e) => {
      el.setPointerCapture(e.pointerId);
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
      moved = 0;
      if (pts.size === 2) {
        const [a, b] = [...pts.values()];
        pinch = Math.hypot(a.x - b.x, a.y - b.y);
      }
    });
    el.addEventListener("pointermove", (e) => {
      const prev = pts.get(e.pointerId);
      if (!prev) return;
      const dx = e.clientX - prev.x;
      const dy = e.clientY - prev.y;
      moved += Math.abs(dx) + Math.abs(dy);
      if (moved < 4) return;
      this.markInput();
      if (pts.size === 2) {
        prev.x = e.clientX;
        prev.y = e.clientY;
        const [a, b] = [...pts.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinch > 0) this.zoom(pinch / d);
        pinch = d;
        this.pan(dx / 2, dy / 2);
        return;
      }
      prev.x = e.clientX;
      prev.y = e.clientY;
      if (e.buttons & 2 || e.shiftKey) this.pan(dx, dy);
      else {
        this.yaw -= dx * 0.005;
        this.pitch = Math.min(1.45, Math.max(0.1, this.pitch + dy * 0.004));
      }
    });
    const up = (e: PointerEvent) => {
      const was = pts.get(e.pointerId);
      pts.delete(e.pointerId);
      if (was && moved < 4 && pts.size === 0 && e.button === 0 && e.type === "pointerup") {
        const hit = this.pick(e.clientX, e.clientY);
        if (hit && this.onPick) this.onPick(hit[0], hit[1]);
      }
    };
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
    el.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        this.markInput();
        this.zoom(Math.exp(e.deltaY * 0.0012));
      },
      { passive: false },
    );
  }

  private zoom(f: number) {
    const max = Math.max(this.sim.width, this.sim.height) * 1.6;
    this.dist = Math.min(max, Math.max(6, this.dist * f));
  }

  private pan(dx: number, dy: number) {
    const k = this.dist * 0.0016;
    const cy = Math.cos(this.yaw);
    const sy = Math.sin(this.yaw);
    this.tx = Math.min(this.sim.width, Math.max(0, this.tx - (dx * cy + dy * sy) * k));
    this.tz = Math.min(this.sim.height, Math.max(0, this.tz - (-dx * sy + dy * cy) * k));
  }

  /** Grid coordinates under a screen point, on the water plane. */
  pick(clientX: number, clientY: number): [number, number] | null {
    if (!this.canvas) return null;
    const r = this.canvas.getBoundingClientRect();
    const x = ((clientX - r.left) / r.width) * 2 - 1;
    const y = 1 - ((clientY - r.top) / r.height) * 2;
    const a = transform(this.invVP, [x, y, 0, 1]);
    const b = transform(this.invVP, [x, y, 1, 1]);
    const dy = b[1] - a[1];
    if (Math.abs(dy) < 1e-6) return null;
    const t = -a[1] / dy;
    if (t < 0) return null;
    const px = a[0] + (b[0] - a[0]) * t;
    const pz = a[2] + (b[2] - a[2]) * t;
    if (px < 0 || pz < 0 || px >= this.sim.width || pz >= this.sim.height) return null;
    return [px, pz];
  }

  private ensureTargets(w: number, h: number) {
    if (this.msaa && this.msaa.width === w && this.msaa.height === h) return;
    this.msaa?.destroy();
    this.depth?.destroy();
    const device = this.sim.device;
    this.msaa = device.createTexture({ size: [w, h], sampleCount: 4, format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.depth = device.createTexture({ size: [w, h], sampleCount: 4, format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  }

  render(frame: Lake3DFrame): void {
    if (!this.canvas || !this.ctx) return;
    const cw = this.canvas.clientWidth;
    const ch = this.canvas.clientHeight;
    if (cw === 0 || ch === 0) return;
    // Adaptive resolution: drop pixels when frames run long, recover when fast.
    const now = performance.now();
    if (now - this.lastScaleChange > 600) {
      if (this.frameMs > 24 && this.scale > 0.5) {
        this.scale = Math.max(0.5, this.scale * 0.85);
        this.lastScaleChange = now;
      } else if (this.frameMs < 14 && this.scale < this.maxScale) {
        this.scale = Math.min(this.maxScale, this.scale * 1.08);
        this.lastScaleChange = now;
      }
    }
    const w = Math.max(1, Math.round(cw * this.scale));
    const h = Math.max(1, Math.round(ch * this.scale));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    this.renderTo(this.ctx.getCurrentTexture().createView(), w, h, frame);
  }

  /** Draw one frame into any texture view of the configured format. */
  renderTo(target: GPUTextureView, w: number, h: number, frame: Lake3DFrame): void {
    this.ensureTargets(w, h);

    const now = performance.now();
    const dt = Math.min(0.1, (now - this.lastFrame) / 1000);
    this.frameMs = this.frameMs * 0.9 + Math.min(100, now - this.lastFrame) * 0.1;
    this.lastFrame = now;
    if (this.autoOrbit && now - this.lastInput > 6000) this.yaw += dt * 0.035;
    if (this.fly) {
      const k = Math.min(1, (now - this.fly.t0) / this.fly.dur);
      const e = k * k * (3 - 2 * k);
      const { from, to } = this.fly;
      this.tx = from[0] + (to[0] - from[0]) * e;
      this.tz = from[1] + (to[1] - from[1]) * e;
      this.dist = from[2] * Math.pow(to[2] / from[2], e);
      if (k >= 1) this.fly = null;
    }

    // Fly-in: start high and far, settle into the orbit.
    let dist = this.dist;
    let pitch = this.pitch;
    const k = this.intro.dur > 0 ? Math.min(1, (now - this.intro.t0) / this.intro.dur) : 1;
    if (k < 1) {
      const e = 1 - Math.pow(1 - k, 3);
      dist = this.dist * (1.9 - 0.9 * e);
      pitch = 1.35 + (this.pitch - 1.35) * e;
    }

    const focus: Vec3 = [this.tx, 0, this.tz];
    const eye: Vec3 = [
      focus[0] + dist * Math.cos(pitch) * Math.sin(this.yaw),
      focus[1] + dist * Math.sin(pitch),
      focus[2] + dist * Math.cos(pitch) * Math.cos(this.yaw),
    ];
    const proj = perspective((38 * Math.PI) / 180, w / h, Math.max(0.05, dist * 0.004), dist * 6 + 3000);
    const vp = mul(proj, lookAt(eye, focus, [0, 1, 0]));
    this.vp = vp;
    this.invVP = invert(vp);

    // Sun from the sim clock: rises ~06:15, sets ~18:05 IST in late November.
    const a = ((frame.hour - 6.25) / 11.85) * Math.PI;
    const sun = norm([Math.cos(a), Math.sin(a) * 0.95, 0.38]);
    const daylight = smooth(-0.14, 0.12, sun[1]);
    const ws = frame.wind.speed;
    const wl = Math.hypot(frame.wind.u, frame.wind.v) || 1;

    const c = this.camData;
    c.set(vp, 0);
    c.set(this.invVP, 16);
    c.set([eye[0], eye[1], eye[2], now / 1000], 32);
    c.set([sun[0], sun[1], sun[2], daylight], 36);
    c.set([frame.wind.u / wl, -frame.wind.v / wl, ws, this.heatOn ? 1 : 0], 40);
    const span = Math.max(this.sim.width, this.sim.height);
    c.set([1.7, 1.15, 0.9 / (span + dist), 0], 44);
    // LOD: plants beyond these distances are skipped (the water tint and
    // terrain carry the picture there).
    c.set([this.stride, this.gridW, 420, 900], 48);
    const device = this.sim.device;
    device.queue.writeBuffer(this.camBuf, 0, c);
    this.sim.writeParams();

    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [
        {
          view: this.msaa!.createView(),
          resolveTarget: target,
          loadOp: "clear",
          storeOp: "discard",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
      depthStencilAttachment: { view: this.depth!.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "discard" },
    });
    pass.setBindGroup(0, this.group);
    pass.setPipeline(this.pipes.terrain);
    pass.setIndexBuffer(this.terrainIndex, "uint32");
    pass.drawIndexed(this.terrainIndexCount);
    pass.setPipeline(this.pipes.skirt);
    pass.draw(24);
    if (this.palmCount) {
      pass.setPipeline(this.pipes.palm);
      pass.draw(6 + 6 * 6, this.palmCount);
    }
    pass.setPipeline(this.pipes.mat);
    pass.draw(30, this.sim.activeParticleSlots);
    // Sky before the (non-depth-writing) water, after everything that writes depth.
    pass.setPipeline(this.pipes.sky);
    pass.draw(3);
    pass.setPipeline(this.pipes.water);
    pass.draw(6);
    pass.end();
    device.queue.submit([enc.finish()]);
  }
}
