// Hyacinth mats as particles. Each particle is vec4(x, y, state, age_steps)
// in cell units; state 0 = dead/empty slot, 1 = drifting, 2 = stuck.
//
// velocity = (current + windage drift) * crowding + cohesion + jitter
//   - current : bilinear sample of the LBM velocity field
//   - windage : mats stick up out of the water, so ~3 % of wind speed
//   - crowding: dense rafts slow down (pile-up at bends and inlets)
//   - cohesion: small pull up the density gradient so mats clump into rafts
// A particle whose next cell is a wall sticks there; stuck mats only break
// free with probability stickRelease per step. Mats that reach a sea outlet
// are flushed out of the domain.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> cellType: array<u32>;
@group(0) @binding(2) var<storage, read> vel: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> density: array<u32>;
@group(0) @binding(4) var<storage, read_write> particles: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> count: array<atomic<u32>>;

fn inGrid(x: i32, y: i32) -> bool {
  return x >= 0 && y >= 0 && x < i32(P.width) && y < i32(P.height);
}

fn velAt(x: i32, y: i32) -> vec2<f32> {
  if (!inGrid(x, y)) {
    return vec2<f32>(0.0, 0.0);
  }
  return vel[cellIndex(u32(x), u32(y), P.width)];
}

fn densAt(x: i32, y: i32) -> f32 {
  if (!inGrid(x, y)) {
    return 0.0;
  }
  return f32(density[cellIndex(u32(x), u32(y), P.width)]);
}

// Bilinear interpolation of the cell-centred velocity field.
fn sampleVel(p: vec2<f32>) -> vec2<f32> {
  let q = p - vec2<f32>(0.5, 0.5);
  let x0 = i32(floor(q.x));
  let y0 = i32(floor(q.y));
  let fx = q.x - f32(x0);
  let fy = q.y - f32(y0);
  let a = mix(velAt(x0, y0), velAt(x0 + 1, y0), fx);
  let b = mix(velAt(x0, y0 + 1), velAt(x0 + 1, y0 + 1), fx);
  return mix(a, b, fy);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.numParticles) {
    return;
  }
  var p = particles[i];
  if (p.z < 0.5) {
    return;
  }

  let cx = i32(floor(p.x));
  let cy = i32(floor(p.y));

  // density includes this particle itself (last step's count), so crowding
  // only starts with company.
  let crowd = 1.0 / (1.0 + max(densAt(cx, cy) - 1.0, 0.0) / P.crowdCap);
  let grad = vec2<f32>(
    densAt(cx + 1, cy) - densAt(cx - 1, cy),
    densAt(cx, cy + 1) - densAt(cx, cy - 1)
  );
  let coh = P.cohesion * grad / (1.0 + length(grad));

  let r1 = rand01(i, P.step * 2u + P.seed);
  let r2 = rand01(i ^ 0x9e3779b9u, P.step * 2u + 1u + P.seed);
  let r3 = rand01(i ^ 0x85ebca6bu, P.step + P.seed * 7u);
  let jit = (vec2<f32>(r1, r2) - vec2<f32>(0.5, 0.5)) * (2.0 * P.jitter);

  let v = (sampleVel(p.xy) + P.windDrift) * crowd + coh + jit;
  let np = p.xy + v;
  let nx = i32(floor(np.x));
  let ny = i32(floor(np.y));

  var blocked = !inGrid(nx, ny);
  var flushed = false;
  if (!blocked) {
    let t = cellType[cellIndex(u32(nx), u32(ny), P.width)];
    blocked = isSolidType(t, P.barrageOpen);
    flushed = t == CELL_OUTLET;
  }

  if (flushed) {
    particles[i] = vec4<f32>(np, 0.0, p.w + 1.0);
    return;
  }

  if (blocked) {
    p.z = 2.0;
  } else if (p.z > 1.5) {
    // Stuck: mats are tangled against the bank and only sometimes let go.
    if (r3 < P.stickRelease) {
      p = vec4<f32>(np, 1.0, p.w);
    }
  } else {
    p = vec4<f32>(np, 1.0, p.w);
  }
  p.w = p.w + 1.0;
  particles[i] = p;

  let fx = u32(clamp(i32(floor(p.x)), 0, i32(P.width) - 1));
  let fy = u32(clamp(i32(floor(p.y)), 0, i32(P.height) - 1));
  atomicAdd(&count[cellIndex(fx, fy, P.width)], 1u);
}
