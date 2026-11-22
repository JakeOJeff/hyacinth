// Shared declarations, prepended to every Hyacinth shader by the engine.
// Keep this file free of @binding declarations: each pipeline uses layout "auto",
// which drops unused bindings, so bindings live in the individual shaders.
//
// Grid convention: x grows east, y grows SOUTH (row 0 is the northern edge),
// matching image rows in mask.png / seed.png.

struct Params {
  width: u32,
  height: u32,
  step: u32,            // global step counter since t0
  numParticles: u32,    // particle buffer capacity (dead slots are skipped)

  tau: f32,             // BGK relaxation time (keep >= 0.55)
  rhoIn: f32,           // density at river-inflow cells (pressure boundary)
  rhoOut: f32,          // density at sea-outlet cells (pressure boundary)
  barrageOpen: u32,     // 1 = Thanneermukkom shutters open, 0 = closed (wall)

  windForce: vec2<f32>, // body acceleration on water per step, lattice units
  drag: f32,            // linear bottom friction per step
  maxU: f32,            // velocity clamp, lattice units (stability guard)

  windDrift: vec2<f32>, // extra mat drift from windage, cells per step
  jitter: f32,          // random walk amplitude, cells per step
  stickRelease: f32,    // per-step probability a stuck mat breaks free

  jamThreshold: f32,    // particles per cell that count as "blocked"
  crowdCap: f32,        // particles per cell where mats start to slow down
  cohesion: f32,        // pull up the density gradient (raft clumping)
  seed: u32,            // RNG seed

  viewMode: u32,        // 0 = live view, 1 = jam heatmap
  showObserved: u32,    // 1 = draw observed (hindcast) mats as outline
  jamUnit: f32,         // dwell (particle-steps) equal to one hour of full jam
  progress: f32,        // 0..1, drawn as a bar at the bottom of the frame

  recordStart: u32,     // jam / ETA accumulate only from this step on
  jamSustain: u32,      // steps a cell must stay blocked to count as jammed
  _pad1: u32,
  _pad2: u32,
};

const CELL_LAND: u32 = 0u;
const CELL_WATER: u32 = 1u;
const CELL_INFLOW: u32 = 2u;
const CELL_OUTLET: u32 = 3u;
const CELL_BARRAGE: u32 = 4u;

const NO_ETA: u32 = 0xffffffffu;

// Tile-list dispatches are laid out (TILE_ROW, ceil(count / TILE_ROW)) so
// big grids stay under the 65535 workgroups-per-dimension limit.
const TILE_ROW: u32 = 32768u;

fn isSolidType(t: u32, barrageOpen: u32) -> bool {
  return t == CELL_LAND || (t == CELL_BARRAGE && barrageOpen == 0u);
}

fn cellIndex(x: u32, y: u32, w: u32) -> u32 {
  return y * w + x;
}

// PCG hash -> uniform float in [0, 1)
fn pcg(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn rand01(a: u32, b: u32) -> f32 {
  return f32(pcg(a ^ pcg(b)) >> 8u) / 16777216.0;
}
