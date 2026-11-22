// Full-screen render of the simulation state. The render target is exactly
// width x height pixels (one pixel per cell); the browser scales it with CSS.
//
// viewMode 0: water shaded by current speed, mats in green by density.
// viewMode 1: jam heatmap (log jam-hours), the daily forecast product.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> cellType: array<u32>;
@group(0) @binding(2) var<storage, read> vel: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> density: array<u32>;
@group(0) @binding(4) var<storage, read> jam: array<u32>;
@group(0) @binding(5) var<storage, read> observed: array<u32>;

struct VOut {
  @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  // One oversized triangle covering the viewport.
  var xy = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0)
  );
  var o: VOut;
  o.pos = vec4<f32>(xy[vi], 0.0, 1.0);
  return o;
}

const LAND = vec3<f32>(0.106, 0.122, 0.110);
const WATER_DEEP = vec3<f32>(0.055, 0.180, 0.247);
const WATER_FAST = vec3<f32>(0.247, 0.522, 0.639);
const MAT = vec3<f32>(0.353, 0.784, 0.329);
const MAT_DENSE = vec3<f32>(0.851, 0.937, 0.373);
const BARRAGE_CLOSED = vec3<f32>(0.937, 0.533, 0.165);
const BARRAGE_OPEN = vec3<f32>(0.620, 0.580, 0.450);
const OBSERVED = vec3<f32>(0.957, 0.318, 0.722);

fn heatRamp(t: f32) -> vec3<f32> {
  // dark water -> amber -> orange -> deep red
  let s0 = WATER_DEEP;
  let s1 = vec3<f32>(0.992, 0.835, 0.400);
  let s2 = vec3<f32>(0.976, 0.451, 0.086);
  let s3 = vec3<f32>(0.725, 0.110, 0.110);
  if (t < 0.33) {
    return mix(s0, s1, t / 0.33);
  }
  if (t < 0.66) {
    return mix(s1, s2, (t - 0.33) / 0.33);
  }
  return mix(s2, s3, clamp((t - 0.66) / 0.34, 0.0, 1.0));
}

fn obsAt(x: i32, y: i32) -> bool {
  if (x < 0 || y < 0 || x >= i32(P.width) || y >= i32(P.height)) {
    return false;
  }
  return observed[cellIndex(u32(x), u32(y), P.width)] > 127u;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
  let x = u32(in.pos.x);
  let y = u32(in.pos.y);
  if (x >= P.width || y >= P.height) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }

  // Progress bar along the bottom 3 rows.
  if (P.progress > 0.0 && y + 3u >= P.height) {
    if (f32(x) / f32(P.width) <= P.progress) {
      return vec4<f32>(0.992, 0.835, 0.400, 1.0);
    }
    return vec4<f32>(0.05, 0.05, 0.05, 1.0);
  }

  let c = cellIndex(x, y, P.width);
  let t = cellType[c];
  var col = LAND;

  if (t == CELL_BARRAGE) {
    if (P.barrageOpen == 1u) {
      col = BARRAGE_OPEN;
    } else {
      col = BARRAGE_CLOSED;
    }
  } else if (t != CELL_LAND) {
    if (P.viewMode == 1u) {
      // Jam-hours at blocking density; 12 h or more is full red. A raft
      // drifting past adds only a few hundredths of an hour and stays dark.
      let jh = f32(jam[c]) / P.jamUnit;
      let tt = clamp(log(1.0 + jh) / log(13.0), 0.0, 1.0);
      col = heatRamp(tt);
    } else {
      let s = clamp(length(vel[c]) / P.maxU, 0.0, 1.0);
      col = mix(WATER_DEEP, WATER_FAST, sqrt(s));
      let d = f32(density[c]);
      if (d > 0.0) {
        let a = clamp(0.35 + 0.65 * d / P.crowdCap, 0.0, 1.0);
        let m = mix(MAT, MAT_DENSE, clamp(d / (2.0 * P.jamThreshold), 0.0, 1.0));
        col = mix(col, m, a);
      }
    }
  }

  if (P.showObserved == 1u) {
    let xi = i32(x);
    let yi = i32(y);
    let here = obsAt(xi, yi);
    let edge = here && (!obsAt(xi + 1, yi) || !obsAt(xi - 1, yi) || !obsAt(xi, yi + 1) || !obsAt(xi, yi - 1));
    if (edge) {
      col = OBSERVED;
    } else if (here) {
      col = mix(col, OBSERVED, 0.18);
    }
  }

  return vec4<f32>(col, 1.0);
}
