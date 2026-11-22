// D2Q9 lattice-Boltzmann, BGK collision, pull-streaming + collide in one pass.
// Distributions are stored structure-of-arrays: f[i * N + cell].
//
// Only 8x8 tiles that contain water are dispatched (most of the grid is
// land); `tiles` lists them as (tx | ty << 16). Land cells are never read by
// streaming (bounce-back reads the cell's own distributions), so they are
// simply left alone.
//
// cellInfo packs, per cell: bits 0-3 cell type, bits 4-12 "source of
// direction i is land / off-grid", bits 13-21 "source of direction i is the
// barrage". One read replaces nine neighbour cellType lookups.
//
// Boundaries:
//   land / closed barrage : half-way bounce-back (no-slip walls)
//   river inflow          : equilibrium at rhoIn  (pressure inlet)
//   sea outlet            : equilibrium at rhoOut (pressure outlet)
//   (both with zero-gradient velocity from the neighbouring interior water)
// Forcing: wind stress minus linear bottom friction, applied with the
// Shan-Chen velocity shift, u_eq = u + tau * a. Velocities are clamped.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> cellType: array<u32>;
@group(0) @binding(2) var<storage, read> fIn: array<f32>;
@group(0) @binding(3) var<storage, read_write> fOut: array<f32>;
@group(0) @binding(4) var<storage, read_write> vel: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> cellInfo: array<u32>;
@group(0) @binding(6) var<storage, read> tiles: array<u32>;

var<private> CX: array<i32, 9> = array<i32, 9>(0, 1, 0, -1, 0, 1, -1, -1, 1);
var<private> CY: array<i32, 9> = array<i32, 9>(0, 0, 1, 0, -1, 1, 1, -1, -1);
var<private> OPP: array<u32, 9> = array<u32, 9>(0u, 3u, 4u, 1u, 2u, 7u, 8u, 5u, 6u);
var<private> W: array<f32, 9> = array<f32, 9>(
  4.0 / 9.0,
  1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
);

fn feq(i: u32, rho: f32, u: vec2<f32>) -> f32 {
  let cu = f32(CX[i]) * u.x + f32(CY[i]) * u.y;
  let uu = dot(u, u);
  return W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu);
}

// Hot path below is unrolled by hand: dynamically indexed private arrays
// tend to be spilled to slow local memory by GPU compilers.
const W0: f32 = 4.0 / 9.0;
const WS: f32 = 1.0 / 9.0;
const WD: f32 = 1.0 / 36.0;

fn fe(wi: f32, rho: f32, cu: f32, usq: f32) -> f32 {
  return wi * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - usq);
}

// Distribution i pulled from the upstream cell (gx - dx, gy - dy), or
// bounced back from this cell's opposite direction if that cell is a wall.
fn pull(i: u32, opp: u32, dx: i32, dy: i32, walls: u32, c: u32, n: u32, gx: u32, gy: u32, w: u32) -> f32 {
  if (((walls >> i) & 1u) != 0u) {
    return fIn[opp * n + c];
  }
  return fIn[i * n + u32(i32(gy) - dy) * w + u32(i32(gx) - dx)];
}

// Mean velocity of the WATER cells around (x, y), from the pre-stream
// distributions of the previous step (fIn is read-only, so no race).
// Only runs on the few inflow / outlet cells.
fn interiorVelocity(x: u32, y: u32, w: u32, h: u32, n: u32) -> vec2<f32> {
  var sum = vec2<f32>(0.0, 0.0);
  var k = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let nx = i32(x) + dx;
      let ny = i32(y) + dy;
      if ((dx == 0 && dy == 0) || nx < 0 || ny < 0 || nx >= i32(w) || ny >= i32(h)) {
        continue;
      }
      let s = cellIndex(u32(nx), u32(ny), w);
      if (cellType[s] != CELL_WATER) {
        continue;
      }
      var rho = 0.0;
      var mom = vec2<f32>(0.0, 0.0);
      for (var i = 0u; i < 9u; i++) {
        let fi = fIn[i * n + s];
        rho += fi;
        mom += vec2<f32>(f32(CX[i]), f32(CY[i])) * fi;
      }
      if (rho > 0.5 && rho < 1.5) {
        sum += mom / rho;
        k += 1.0;
      }
    }
  }
  if (k > 0.0) {
    return sum / k;
  }
  return vec2<f32>(0.0, 0.0);
}

fn clampVel(u: vec2<f32>, m: f32) -> vec2<f32> {
  let s = length(u);
  if (s > m) {
    return u * (m / s);
  }
  return u;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let ti = wid.x + wid.y * TILE_ROW;
  if (ti >= arrayLength(&tiles)) {
    return;
  }
  let tile = tiles[ti];
  let gx = (tile & 0xffffu) * 8u + lid.x;
  let gy = (tile >> 16u) * 8u + lid.y;
  let w = P.width;
  let h = P.height;
  if (gx >= w || gy >= h) {
    return;
  }
  let n = w * h;
  let c = cellIndex(gx, gy, w);
  let info = cellInfo[c];
  let t = info & 15u;
  let closed = select(1u, 0u, P.barrageOpen == 1u);

  if (t == CELL_LAND) {
    return;
  }
  if (t == CELL_BARRAGE && closed == 1u) {
    // Keep a closed barrage at rest equilibrium so it opens from a sane state.
    for (var i = 0u; i < 9u; i++) {
      fOut[i * n + c] = W[i];
    }
    vel[c] = vec2<f32>(0.0, 0.0);
    return;
  }

  // Pull streaming with half-way bounce-back.
  let walls = ((info >> 4u) & 0x1ffu) | (((info >> 13u) & 0x1ffu) * closed);
  let f0 = fIn[c];
  let f1 = pull(1u, 3u, 1, 0, walls, c, n, gx, gy, w);
  let f2 = pull(2u, 4u, 0, 1, walls, c, n, gx, gy, w);
  let f3 = pull(3u, 1u, -1, 0, walls, c, n, gx, gy, w);
  let f4 = pull(4u, 2u, 0, -1, walls, c, n, gx, gy, w);
  let f5 = pull(5u, 7u, 1, 1, walls, c, n, gx, gy, w);
  let f6 = pull(6u, 8u, -1, 1, walls, c, n, gx, gy, w);
  let f7 = pull(7u, 5u, -1, -1, walls, c, n, gx, gy, w);
  let f8 = pull(8u, 6u, 1, -1, walls, c, n, gx, gy, w);

  var rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
  let mom = vec2<f32>(f1 - f3 + f5 - f6 - f7 + f8, f2 - f4 + f5 + f6 - f7 - f8);

  // Stability guard: a cell that has gone bad is reset to rest.
  let healthy = rho > 0.5 && rho < 1.5;
  var u = vec2<f32>(0.0, 0.0);
  if (healthy) {
    u = clampVel(mom / rho, P.maxU);
  } else {
    rho = 1.0;
  }

  if (t == CELL_INFLOW || t == CELL_OUTLET) {
    // Pressure boundary: fixed density, zero-gradient velocity taken from the
    // neighbouring interior water (previous step), so fluid and momentum can
    // actually leave or enter the domain here.
    var rb = P.rhoIn;
    if (t == CELL_OUTLET) {
      rb = P.rhoOut;
    }
    let ub = clampVel(interiorVelocity(gx, gy, w, h, n), P.maxU);
    for (var i = 0u; i < 9u; i++) {
      fOut[i * n + c] = feq(i, rb, ub);
    }
    vel[c] = ub;
    return;
  }

  // Wind stress minus bottom friction (acceleration per step).
  let a = P.windForce - P.drag * u;
  let ueq = clampVel(u + P.tau * a, P.maxU);
  let omega = 1.0 / P.tau;
  let ux = ueq.x;
  let uy = ueq.y;
  let usq = 1.5 * (ux * ux + uy * uy);
  fOut[c] = f0 - omega * (f0 - fe(W0, rho, 0.0, usq));
  fOut[n + c] = f1 - omega * (f1 - fe(WS, rho, ux, usq));
  fOut[2u * n + c] = f2 - omega * (f2 - fe(WS, rho, uy, usq));
  fOut[3u * n + c] = f3 - omega * (f3 - fe(WS, rho, -ux, usq));
  fOut[4u * n + c] = f4 - omega * (f4 - fe(WS, rho, -uy, usq));
  fOut[5u * n + c] = f5 - omega * (f5 - fe(WD, rho, ux + uy, usq));
  fOut[6u * n + c] = f6 - omega * (f6 - fe(WD, rho, uy - ux, usq));
  fOut[7u * n + c] = f7 - omega * (f7 - fe(WD, rho, -ux - uy, usq));
  fOut[8u * n + c] = f8 - omega * (f8 - fe(WD, rho, ux - uy, usq));
  // Physical (half-step) velocity for the forced scheme.
  vel[c] = clampVel(u + 0.5 * a, P.maxU);
}
