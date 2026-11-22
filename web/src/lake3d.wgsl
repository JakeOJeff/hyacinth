// 3D view of the live simulation. Prepended with sim/shaders/common.wgsl
// (Params, cell types, pcg / rand01). Everything reads the simulation's own
// GPU buffers directly: no CPU readback per frame.
//
// World space: x = grid x (east), z = grid y (south), y = up. One unit = one
// cell (40 m). Heights are exaggerated so banks and embankments read.

struct Cam {
  viewProj: mat4x4<f32>,
  invViewProj: mat4x4<f32>,
  eye: vec4<f32>,   // xyz eye, w = wall-clock seconds (animation)
  sun: vec4<f32>,   // xyz direction to the sun, w = daylight 0..1
  wind: vec4<f32>,  // xy = wind blowing-to direction in world xz, z = speed m/s, w = jam glow on
  misc: vec4<f32>,  // x = hyacinth size, y = palm height, z = fog density, w = unused
  lod: vec4<f32>,   // x = terrain stride (cells), y = terrain grid width, z = hyacinth cull distance, w = palm cull distance
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> C: Cam;
@group(0) @binding(2) var<storage, read> cellType: array<u32>;
@group(0) @binding(3) var<storage, read> vel: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> density: array<u32>;
@group(0) @binding(5) var<storage, read> jam: array<u32>;
@group(0) @binding(6) var<storage, read> particles: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> heights: array<f32>;
@group(0) @binding(8) var<storage, read> palms: array<vec4<f32>>;

const UP = vec3<f32>(0.0, 1.0, 0.0);
const TAU = 6.2831853;

fn srgb(c: vec3<f32>) -> vec3<f32> {
  return pow(c, vec3<f32>(2.2));
}

fn hashI(p: vec2<f32>) -> f32 {
  let x = bitcast<u32>(i32(p.x));
  let y = bitcast<u32>(i32(p.y));
  return f32(pcg(x ^ pcg(y + 0x9e3779b9u)) >> 8u) / 16777216.0;
}

fn vnoise(p: vec2<f32>) -> f32 {
  let i = floor(p);
  let f = p - i;
  let u = f * f * (3.0 - 2.0 * f);
  let a = hashI(i);
  let b = hashI(i + vec2<f32>(1.0, 0.0));
  let c = hashI(i + vec2<f32>(0.0, 1.0));
  let d = hashI(i + vec2<f32>(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Analytic gradient of value noise.
fn vgrad(p: vec2<f32>) -> vec2<f32> {
  let i = floor(p);
  let f = p - i;
  let u = f * f * (3.0 - 2.0 * f);
  let du = 6.0 * f * (1.0 - f);
  let a = hashI(i);
  let b = hashI(i + vec2<f32>(1.0, 0.0));
  let c = hashI(i + vec2<f32>(0.0, 1.0));
  let d = hashI(i + vec2<f32>(1.0, 1.0));
  let k = a - b - c + d;
  return vec2<f32>(du.x * ((b - a) + k * u.y), du.y * ((c - a) + k * u.x));
}

fn cellOf(x: i32, y: i32) -> u32 {
  let xi = clamp(x, 0, i32(P.width) - 1);
  let yi = clamp(y, 0, i32(P.height) - 1);
  return cellIndex(u32(xi), u32(yi), P.width);
}

// ---------- lighting ----------

const MOON_DIR = vec3<f32>(-0.32, 0.82, 0.47);

fn sunColor(y: f32) -> vec3<f32> {
  return mix(srgb(vec3<f32>(1.0, 0.50, 0.24)) * 1.8, vec3<f32>(1.0, 0.95, 0.88) * 2.6, smoothstep(0.0, 0.45, y));
}

fn lightDir() -> vec3<f32> {
  return normalize(mix(normalize(MOON_DIR), normalize(C.sun.xyz), C.sun.w));
}

fn lightCol() -> vec3<f32> {
  return mix(srgb(vec3<f32>(0.55, 0.66, 0.95)) * 0.95, sunColor(C.sun.y), C.sun.w);
}

fn ambient(n: vec3<f32>) -> vec3<f32> {
  let sky = mix(srgb(vec3<f32>(0.22, 0.28, 0.46)) * 1.1, srgb(vec3<f32>(0.50, 0.62, 0.80)) * 0.75, C.sun.w);
  let ground = mix(srgb(vec3<f32>(0.06, 0.07, 0.10)), srgb(vec3<f32>(0.30, 0.32, 0.22)) * 0.35, C.sun.w);
  return mix(ground, sky, n.y * 0.5 + 0.5);
}

fn shade(albedo: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
  let diff = max(dot(n, lightDir()), 0.0) * lightCol();
  return albedo * (diff + ambient(n));
}

fn sky(rd0: vec3<f32>) -> vec3<f32> {
  let rd = normalize(vec3<f32>(rd0.x, max(rd0.y, -0.02), rd0.z));
  let d = C.sun.w;
  let s = normalize(C.sun.xyz);
  let up = clamp(rd.y, 0.0, 1.0);
  let toward = pow(max(dot(rd, s) * 0.5 + 0.5, 0.0), 6.0);
  let dusk = clamp(d * (1.0 - smoothstep(0.02, 0.42, s.y)) * 1.4, 0.0, 1.0);
  let dayHor = mix(srgb(vec3<f32>(0.84, 0.82, 0.77)), srgb(vec3<f32>(1.0, 0.56, 0.30)), dusk * (0.35 + 0.65 * toward));
  let dayZen = mix(srgb(vec3<f32>(0.42, 0.56, 0.72)), srgb(vec3<f32>(0.22, 0.24, 0.50)), dusk * 0.7);
  let hor = mix(srgb(vec3<f32>(0.14, 0.19, 0.32)), dayHor, d);
  let zen = mix(srgb(vec3<f32>(0.02, 0.03, 0.08)), dayZen, d);
  var col = mix(hor, zen, pow(up, 0.45));
  let sd = max(dot(rd, s), 0.0);
  col += sunColor(s.y) * (pow(sd, 1500.0) * 40.0 + pow(sd, 10.0) * 0.18) * smoothstep(-0.05, 0.05, s.y);
  let night = 1.0 - d;
  if (night > 0.05 && rd.y > 0.0) {
    let g = floor(rd * 300.0);
    let h = hashI(vec2<f32>(g.x + g.z * 613.0, g.y));
    col += vec3<f32>(smoothstep(0.9965, 1.0, h)) * night * 2.0 * smoothstep(0.0, 0.25, up);
  }
  let md = max(dot(rd, normalize(MOON_DIR)), 0.0);
  col += vec3<f32>(0.85, 0.9, 1.0) * (pow(md, 3000.0) * 8.0 + pow(md, 40.0) * 0.05) * night;
  return col;
}

// Sky colour at the horizon only: the cheap part of sky(), used for fog.
fn horizonColor(rd: vec3<f32>) -> vec3<f32> {
  let d = C.sun.w;
  let s = normalize(C.sun.xyz);
  let toward = pow(max(dot(rd, s) * 0.5 + 0.5, 0.0), 6.0);
  let dusk = clamp(d * (1.0 - smoothstep(0.02, 0.42, s.y)) * 1.4, 0.0, 1.0);
  let dayHor = mix(srgb(vec3<f32>(0.84, 0.82, 0.77)), srgb(vec3<f32>(1.0, 0.56, 0.30)), dusk * (0.35 + 0.65 * toward));
  return mix(srgb(vec3<f32>(0.14, 0.19, 0.32)), dayHor, d);
}

fn fogged(col: vec3<f32>, world: vec3<f32>) -> vec3<f32> {
  let dv = world - C.eye.xyz;
  let f = 1.0 - exp(-length(dv) * C.misc.z);
  let fc = horizonColor(normalize(vec3<f32>(dv.x, 0.0, dv.z)));
  return mix(col, fc, f * 0.9);
}

fn tonemap(c: vec3<f32>) -> vec3<f32> {
  let x = c * 0.85;
  let a = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
  return pow(clamp(a, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
}

fn heatRamp(t: f32) -> vec3<f32> {
  let s0 = vec3<f32>(0.992, 0.835, 0.400);
  let s1 = vec3<f32>(0.976, 0.451, 0.086);
  let s2 = vec3<f32>(0.85, 0.10, 0.10);
  if (t < 0.5) {
    return mix(s0, s1, t / 0.5);
  }
  return mix(s1, s2, (t - 0.5) / 0.5);
}

// ---------- sky ----------

struct SOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) ndc: vec2<f32>,
};

@vertex
fn skyVs(@builtin(vertex_index) vi: u32) -> SOut {
  var xy = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var o: SOut;
  o.pos = vec4<f32>(xy[vi], 1.0, 1.0);
  o.ndc = xy[vi];
  return o;
}

@fragment
fn skyFs(i: SOut) -> @location(0) vec4<f32> {
  let a = C.invViewProj * vec4<f32>(i.ndc, 1.0, 1.0);
  let rd = normalize(a.xyz / a.w - C.eye.xyz);
  return vec4<f32>(tonemap(sky(rd)), 1.0);
}

// ---------- terrain ----------

fn hAt(x: i32, y: i32) -> f32 {
  let c = cellOf(x, y);
  if (cellType[c] == CELL_BARRAGE) {
    return select(0.9, -0.4, P.barrageOpen == 1u);
  }
  return heights[c];
}

struct TOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) n: vec3<f32>,
  @location(2) bar: f32,
};

// One vertex per grid point (indexed draw), every lod.x cells.
@vertex
fn terrainVs(@builtin(vertex_index) vi: u32) -> TOut {
  let S = i32(C.lod.x);
  let gw = u32(C.lod.y);
  let x = min(i32(vi % gw) * S, i32(P.width) - 1);
  let y = min(i32(vi / gw) * S, i32(P.height) - 1);
  // A thin barrage line can fall between samples: look across the whole block.
  var bar = false;
  for (var j = 0; j < S; j++) {
    for (var i = 0; i < S; i++) {
      if (cellType[cellOf(x + i, y + j)] == CELL_BARRAGE) {
        bar = true;
      }
    }
  }
  var h = hAt(x, y);
  if (bar) {
    h = select(0.9, -0.4, P.barrageOpen == 1u);
  }
  let dx = hAt(x + S, y) - hAt(x - S, y);
  let dz = hAt(x, y + S) - hAt(x, y - S);
  var o: TOut;
  o.world = vec3<f32>(f32(x) + 0.5, h, f32(y) + 0.5);
  o.pos = C.viewProj * vec4<f32>(o.world, 1.0);
  o.n = normalize(vec3<f32>(-dx * 0.9 / f32(S), 1.0, -dz * 0.9 / f32(S)));
  o.bar = select(0.0, 1.0, bar);
  return o;
}

@fragment
fn terrainFs(i: TOut) -> @location(0) vec4<f32> {
  let h = i.world.y;
  let p = i.world.xz;
  let dist = length(i.world - C.eye.xyz);

  // Paddy-field patchwork (Kuttanad polders), with thin earthen ridges.
  let warp = vec2<f32>(vnoise(p * 0.04), vnoise(p * 0.04 + 7.3)) * 7.0;
  let fq = (p + warp) / vec2<f32>(9.0, 6.0);
  let r = hashI(floor(fq));
  var paddy = srgb(vec3<f32>(0.44, 0.52, 0.25));
  if (r > 0.78) {
    paddy = srgb(vec3<f32>(0.64, 0.58, 0.38));
  } else if (r > 0.52) {
    paddy = srgb(vec3<f32>(0.52, 0.58, 0.31));
  } else if (r > 0.40) {
    paddy = srgb(vec3<f32>(0.32, 0.39, 0.27));
  }
  paddy *= 0.82 + 0.18 * vnoise(p * 1.7 + r * 11.0);
  let ff = fract(fq);
  let edge = min(min(ff.x, 1.0 - ff.x) * 9.0, min(ff.y, 1.0 - ff.y) * 6.0);
  let ridge = 0.55 * (1.0 - smoothstep(0.04, 0.14, edge)) * (1.0 - smoothstep(80.0, 260.0, dist));
  paddy = mix(paddy, srgb(vec3<f32>(0.42, 0.37, 0.27)), ridge);
  let bund = smoothstep(0.26, 0.33, h);
  var land = mix(paddy, srgb(vec3<f32>(0.34, 0.37, 0.21)), bund);
  land = mix(srgb(vec3<f32>(0.24, 0.20, 0.13)), land, smoothstep(0.0, 0.07, h));
  let bed = srgb(vec3<f32>(0.17, 0.15, 0.10)) * (0.65 + 0.35 * vnoise(p * 0.6));
  var alb = select(land, bed, h < 0.0);

  let closed = P.barrageOpen == 0u;
  if (i.bar > 0.35) {
    alb = select(srgb(vec3<f32>(0.58, 0.58, 0.55)), srgb(vec3<f32>(0.97, 0.56, 0.12)), closed);
  }
  var col = shade(alb, normalize(i.n));
  if (i.bar > 0.35 && closed) {
    col += srgb(vec3<f32>(0.97, 0.50, 0.10)) * 0.5;
  }
  return vec4<f32>(tonemap(fogged(col, i.world)), 1.0);
}

// Flat land skirt around the grid so the map has no hard edge.
@vertex
fn skirtVs(@builtin(vertex_index) vi: u32) -> TOut {
  let W = f32(P.width);
  let H = f32(P.height);
  let E = max(W, H) * 4.0;
  var rects = array<vec4<f32>, 4>(
    vec4<f32>(-E, -E, 0.0, H + E), vec4<f32>(W, -E, W + E, H + E),
    vec4<f32>(0.0, -E, W, 0.0), vec4<f32>(0.0, H, W, H + E)
  );
  var c = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
  );
  let r = rects[vi / 6u];
  let k = c[vi % 6u];
  var o: TOut;
  o.world = vec3<f32>(mix(r.x, r.z, k.x), 0.16, mix(r.y, r.w, k.y));
  o.pos = C.viewProj * vec4<f32>(o.world, 1.0);
  o.n = UP;
  o.bar = 0.0;
  return o;
}

// ---------- water ----------

struct WOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) world: vec3<f32>,
};

@vertex
fn waterVs(@builtin(vertex_index) vi: u32) -> WOut {
  var c = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0)
  );
  let p = c[vi] * vec2<f32>(f32(P.width), f32(P.height));
  var o: WOut;
  o.world = vec3<f32>(p.x, 0.0, p.y);
  o.pos = C.viewProj * vec4<f32>(o.world, 1.0);
  return o;
}

fn velCell(x: i32, y: i32) -> vec2<f32> {
  if (x < 0 || y < 0 || x >= i32(P.width) || y >= i32(P.height)) {
    return vec2<f32>(0.0);
  }
  let c = cellIndex(u32(x), u32(y), P.width);
  if (cellType[c] == CELL_LAND) {
    return vec2<f32>(0.0);
  }
  return vel[c];
}

fn velAt(p: vec2<f32>) -> vec2<f32> {
  let q = p - 0.5;
  let i = floor(q);
  let f = q - i;
  let x = i32(i.x);
  let y = i32(i.y);
  return mix(mix(velCell(x, y), velCell(x + 1, y), f.x), mix(velCell(x, y + 1), velCell(x + 1, y + 1), f.x), f.y);
}

@fragment
fn waterFs(i: WOut) -> @location(0) vec4<f32> {
  let p = i.world.xz;
  let c = cellOf(i32(p.x), i32(p.y));
  let hb = heights[c];
  if (hb > 0.05 && cellType[c] != CELL_BARRAGE) {
    discard;
  }
  let t = C.eye.w;
  let v = velAt(p);
  let s = clamp(length(v) / P.maxU, 0.0, 1.0);
  let flow = v / P.maxU;

  // Two-phase flow map: ripples advect along the simulated current.
  let ph0 = fract(t * 0.18);
  let ph1 = fract(t * 0.18 + 0.5);
  let w0 = 1.0 - abs(2.0 * ph0 - 1.0);
  var g = vgrad(p * 0.8 - flow * ph0 * 3.0) * w0 + vgrad(p * 0.8 + vec2<f32>(0.37, 0.71) - flow * ph1 * 3.0) * (1.0 - w0);
  g += vgrad(p * 3.3 - C.wind.xy * t * 0.9) * (0.2 + 0.07 * C.wind.z);
  g += vgrad(p * 0.18 + vec2<f32>(t * 0.03, -t * 0.02)) * 0.5;
  let amp = 0.05 + 0.08 * s + 0.012 * C.wind.z;
  let n = normalize(vec3<f32>(-g.x * amp, 1.0, -g.y * amp));

  let V = normalize(C.eye.xyz - i.world);
  let ndv = max(dot(n, V), 0.0);
  let fres = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);
  let R = reflect(-V, n);
  let refl = sky(vec3<f32>(R.x, max(R.y, 0.02), R.z));
  let L = lightDir();
  let base = mix(srgb(vec3<f32>(0.10, 0.23, 0.23)), srgb(vec3<f32>(0.17, 0.35, 0.32)), sqrt(s));
  let body = base * (ambient(UP) + lightCol() * 0.3 * max(L.y, 0.0));
  let spec = lightCol() * pow(max(dot(R, L), 0.0), 280.0) * 3.5;
  var col = mix(body, refl * 0.8, fres * 0.75) + spec;

  // Mat carpets tint the surface so they read from far away (plants take over up close).
  let md = f32(density[c]);
  let matA = clamp(md / max(P.crowdCap, 1.0), 0.0, 1.0) * 0.85;
  let matCol = srgb(mix(vec3<f32>(0.26, 0.46, 0.16), vec3<f32>(0.48, 0.58, 0.22), clamp(md / (3.0 * P.jamThreshold), 0.0, 1.0)));
  col = mix(col, shade(matCol, UP), matA);

  // Foam lines along the banks.
  let shore = smoothstep(-0.42, -0.12, hb);
  let foam = shore * smoothstep(0.62, 0.9, vnoise(p * 2.5 - flow * t * 1.5 + vec2<f32>(t * 0.25, 0.0))) * 0.06;
  col += foam * (ambient(UP) + lightCol() * 0.5);

  var alpha = max(max(mix(0.6, 0.94, smoothstep(0.0, 0.7, -hb)), fres), matA);

  // Jam glow: hours this cell has stayed blocked since recording began.
  if (C.wind.w > 0.5) {
    let jh = f32(jam[c]) / P.jamUnit;
    let tt = clamp(log(1.0 + jh) / log(13.0), 0.0, 1.0);
    let on = smoothstep(0.02, 0.10, tt);
    col = mix(col, srgb(heatRamp(tt)) * (2.0 + 0.4 * sin(t * 2.6)), on * 0.9);
    alpha = mix(alpha, 0.97, on);
  }
  return vec4<f32>(tonemap(fogged(col, i.world)), alpha);
}

// ---------- hyacinth & palms ----------

struct MOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) n: vec3<f32>,
  @location(2) col: vec3<f32>,
  @location(3) gloss: f32,
};

fn offscreen() -> MOut {
  var o: MOut;
  o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  o.world = vec3<f32>(0.0);
  o.n = UP;
  o.col = vec3<f32>(0.0);
  o.gloss = 0.0;
  return o;
}

@vertex
fn matVs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> MOut {
  let pt = particles[ii];
  if (pt.z < 0.5) {
    return offscreen();
  }
  if (distance(vec2<f32>(pt.x, pt.y), C.eye.xz) > C.lod.z + C.eye.y) {
    return offscreen();
  }
  let r0 = rand01(ii, 11u);
  let r1 = rand01(ii, 23u);
  let r2 = rand01(ii, 37u);
  let t = C.eye.w;
  let d = f32(density[cellOf(i32(pt.x), i32(pt.y))]);
  let crowd = clamp(d / max(P.crowdCap, 1.0), 0.0, 1.5);
  let L = C.misc.x * (0.42 + 0.22 * r1) * (1.0 + 0.2 * crowd);
  let base = vec3<f32>(pt.x, 0.02 + 0.012 * sin(t * 1.6 + r0 * TAU), pt.y);

  var lp: vec3<f32>;
  var n: vec3<f32>;
  var col: vec3<f32>;
  var gloss = 1.0;
  if (vi < 24u) {
    let leaf = vi / 6u;
    let j = vi % 6u;
    let a = r0 * TAU + f32(leaf) * 1.5708 + 0.12 * sin(t * 0.7 + r1 * 6.0);
    let dir = vec3<f32>(cos(a), 0.0, sin(a));
    let perp = vec3<f32>(-sin(a), 0.0, cos(a));
    let B = UP * 0.03 * L;
    let Lp = dir * 0.45 * L + perp * 0.26 * L + UP * 0.16 * L;
    let Rp = dir * 0.45 * L - perp * 0.26 * L + UP * 0.16 * L;
    let Tp = dir * 0.95 * L + UP * 0.10 * L;
    var pts = array<vec3<f32>, 6>(B, Lp, Tp, B, Tp, Rp);
    lp = pts[j];
    n = normalize(cross(Tp - B, Lp - Rp));
    if (n.y < 0.0) {
      n = -n;
    }
    let v = 0.78 + 0.4 * rand01(ii, 50u + leaf);
    col = srgb(mix(vec3<f32>(0.20, 0.44, 0.14), vec3<f32>(0.36, 0.56, 0.18), r2)) * v;
    // Piled-up rafts go yellow-green, like the 2D view's dense mats.
    col = mix(col, srgb(vec3<f32>(0.55, 0.62, 0.24)), clamp(d / (4.0 * P.jamThreshold) - 0.5, 0.0, 0.3));
  } else {
    // Lavender flower spike on about one plant in five.
    if (r2 > 0.12) {
      return offscreen();
    }
    let j = vi - 24u;
    let toEye = C.eye.xyz - base;
    let right = normalize(vec3<f32>(-toEye.z, 0.0, toEye.x));
    var q = array<vec2<f32>, 6>(
      vec2<f32>(-1.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(-1.0, 1.0),
      vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
    );
    let c2 = q[j];
    lp = right * c2.x * 0.06 * L + UP * (0.06 * L + c2.y * 0.3 * L);
    n = normalize(normalize(vec3<f32>(toEye.x, 0.0, toEye.z)) + UP * 0.8);
    col = mix(srgb(vec3<f32>(0.42, 0.34, 0.66)), srgb(vec3<f32>(0.62, 0.52, 0.84)), c2.y);
    gloss = 0.2;
  }
  var o: MOut;
  o.world = base + lp;
  o.pos = C.viewProj * vec4<f32>(o.world, 1.0);
  o.n = n;
  o.col = col;
  o.gloss = gloss;
  return o;
}

@vertex
fn palmVs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> MOut {
  let pm = palms[ii]; // x, z, height scale, random
  if (distance(pm.xy, C.eye.xz) > C.lod.w + C.eye.y) {
    return offscreen();
  }
  let t = C.eye.w;
  let ground = max(heights[cellOf(i32(pm.x), i32(pm.y))], 0.0);
  let H = C.misc.y * pm.z;
  let r = pm.w;
  let la = r * TAU;
  let lean = vec3<f32>(cos(la), 0.0, sin(la)) * 0.16 * H;
  let gust = 0.6 + 0.4 * sin(t * 1.3 + r * 20.0);
  let sway = vec3<f32>(C.wind.x, 0.0, C.wind.y) * (0.015 + 0.012 * C.wind.z) * H * gust;
  let base = vec3<f32>(pm.x, ground, pm.y);
  let crown = base + UP * H + lean + sway;

  var o: MOut;
  o.gloss = 0.0;
  if (vi < 6u) {
    let axis = crown - base;
    let toEye = C.eye.xyz - base;
    let side = normalize(cross(axis, toEye));
    let wb = 0.035 * H;
    let wt = 0.022 * H;
    var pts = array<vec3<f32>, 6>(
      base - side * wb, base + side * wb, crown + side * wt,
      base - side * wb, crown + side * wt, crown - side * wt
    );
    o.world = pts[vi];
    o.n = normalize(vec3<f32>(toEye.x, 0.3, toEye.z));
    o.col = srgb(vec3<f32>(0.40, 0.31, 0.22));
  } else {
    let k = vi - 6u;
    let f = k / 6u;
    let j = k % 6u;
    let a = r * 12.0 + f32(f) * 1.0472 + 0.08 * sin(t * 1.7 + f32(f) + r * 9.0);
    let dir = vec3<f32>(cos(a), 0.0, sin(a));
    let perp = vec3<f32>(-sin(a), 0.0, cos(a));
    let Lf = 0.55 * H;
    let droop = 0.26 + 0.14 * fract(r * 7.0 + f32(f) * 0.31);
    let mid = crown + dir * 0.5 * Lf + UP * 0.1 * Lf;
    let tip = crown + dir * Lf - UP * droop * Lf + sway * 0.5;
    let ml = mid + perp * 0.13 * Lf - UP * 0.03 * Lf;
    let mr = mid - perp * 0.13 * Lf - UP * 0.03 * Lf;
    var pts = array<vec3<f32>, 6>(crown, ml, mr, ml, tip, mr);
    o.world = pts[j];
    var n = normalize(cross(ml - crown, mr - crown));
    if (n.y < 0.0) {
      n = -n;
    }
    o.n = n;
    let tipness = select(0.0, 1.0, j == 4u);
    o.col = mix(srgb(vec3<f32>(0.15, 0.34, 0.09)), srgb(vec3<f32>(0.45, 0.55, 0.17)), tipness) * (0.85 + 0.3 * fract(r * 13.0));
  }
  o.pos = C.viewProj * vec4<f32>(o.world, 1.0);
  return o;
}

@fragment
fn plantFs(i: MOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
  var n = normalize(i.n);
  let V = normalize(C.eye.xyz - i.world);
  if (dot(n, V) < 0.0) {
    n = -n;
  }
  var col = shade(i.col, n);
  let H = normalize(lightDir() + V);
  col += lightCol() * pow(max(dot(n, H), 0.0), 60.0) * 0.25 * i.gloss;
  return vec4<f32>(tonemap(fogged(col, i.world)), 1.0);
}
