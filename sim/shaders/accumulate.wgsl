// Per-cell pass after the particle step:
//   density <- this step's particle count (read by the next particle step
//              and by the renderer), count <- 0
//   jam     += density                 (dwell time, in particle-steps)
//   run      = consecutive steps with density >= jamThreshold, counted only
//              in the shore band (cells near a bank, bend, inlet or the
//              barrage), where mats pile up and block a channel
//   eta      = step (relative to recordStart) at which that blockage began,
//              once it has lasted jamSustain steps. A raft drifting across
//              open water shows in the dwell heatmap but is not a "jam".
// Jam and ETA only accumulate once step >= recordStart, so a spin-up from
// the satellite scene time to "now" does not count towards the forecast.

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> count: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> density: array<u32>;
@group(0) @binding(3) var<storage, read_write> jam: array<u32>;
@group(0) @binding(4) var<storage, read_write> eta: array<u32>;
@group(0) @binding(5) var<storage, read_write> run: array<u32>;
@group(0) @binding(6) var<storage, read> shore: array<u32>;
@group(0) @binding(7) var<storage, read> tiles: array<u32>;

// Runs over the same water-tile list as the LBM pass: mats never sit on land.
@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let ti = wid.x + wid.y * TILE_ROW;
  if (ti >= arrayLength(&tiles)) {
    return;
  }
  let tile = tiles[ti];
  let gx = (tile & 0xffffu) * 8u + lid.x;
  let gy = (tile >> 16u) * 8u + lid.y;
  if (gx >= P.width || gy >= P.height) {
    return;
  }
  let c = cellIndex(gx, gy, P.width);
  let d = atomicExchange(&count[c], 0u);
  density[c] = d;
  if (P.step < P.recordStart) {
    return;
  }
  jam[c] = jam[c] + d;
  var r = 0u;
  if (f32(d) >= P.jamThreshold && shore[c] != 0u) {
    r = run[c] + 1u;
  }
  run[c] = r;
  if (r >= P.jamSustain && eta[c] == NO_ETA) {
    // Report the time the blockage began, not when it was confirmed.
    let began = P.step + 1u - min(r, P.step + 1u);
    eta[c] = select(0u, began - P.recordStart, began > P.recordStart);
  }
}
