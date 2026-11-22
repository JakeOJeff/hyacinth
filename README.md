# Hyacinth: Drift Forecast

A free, daily 72-hour forecast of where water hyacinth is likely to jam
Vembanad lake and the Kuttanad canals. It runs a GPU fluid simulation seeded from satellite
imagery and is delivered as a timelapse with a Malayalam + English caption.
Built for the Alappuzha Challenge (Place & People).

**Who it is for.** Clearance crews and panchayats deciding where to send people and harvesters
before mats lock in; boat, ferry and houseboat operators choosing routes; irrigation staff
weighing what the Thanneermukkom barrage does. The site answers their questions directly through
**Ask Hyacinth**, an assistant grounded in the day's run that can re-run the simulation as a what-if.

> It forecasts a **drift likelihood**, not ground truth. The simulation is uncalibrated.

```
Sentinel-2 L2A ─┐
OSM water/canals ┤                       ┌─ browser demo (Vite + WebGPU) ─ Cloudflare Pages
Open-Meteo wind ─┼─ Python pipeline ─────┤
Barrage flag ────┘  mask.png seed.png    └─ Deno headless render ─ MP4 + heatmap + jam stats
                    wind.json                    └─ facts (code) ─ AI caption ─ Telegram
```

The simulation is written once, in WGSL (`sim/shaders/`), and the same engine (`sim/engine.ts`)
runs in the browser and headless in Deno.

## Quick start

Copy `.env.example` to `.env` and fill in your OpenAI key and model (and Telegram, if posting).

Requires Python 3.11+, Node 20+, Deno 2.x and, for the MP4, ffmpeg on `PATH` (or `FFMPEG=...`).

```sh
python -m venv .venv && .venv/Scripts/pip install -r pipeline/requirements.txt   # (bin/ on Linux/macOS)
cd web && npm install && cd ..

deno task selftest                          # GPU physics checks (needs a WebGPU adapter)

cd pipeline
python -m hyacinth synth                       # offline synthetic test domain -> data/synthetic
python -m hyacinth render ../data/synthetic    # 15 s MP4, heatmap, jam/eta rasters
python -m hyacinth caption ../data/synthetic --no-model
python -m hyacinth publish ../data/synthetic synthetic
cd ../web && npm run dev                    # http://localhost:5173
```

Real data:

```sh
cd pipeline
python -m hyacinth fetch                       # OSM mask (cached), newest clear Sentinel-2 scene, wind
python -m hyacinth render ../data/latest
python -m hyacinth caption ../data/latest      # uses OPENAI_MODEL if set, else a template
python -m hyacinth publish ../data/latest latest
python -m hyacinth post ../data/latest         # needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
# or all of it:
python -m hyacinth daily
```

Hindcast (the demo): pick two clear scenes about 5 days apart in Dec to Mar, run the sim
from day 0 with the real historical wind, and overlay the day-5 observation.

```sh
python -m hyacinth hindcast-dates 2025-12-01 2026-03-31
python -m hyacinth hindcast 2026-01-10 2026-01-15
python -m hyacinth render ../data/hindcast
python -m hyacinth score ../data/hindcast      # model vs "mats stay where they were"
python -m hyacinth publish ../data/hindcast hindcast
```

## The contract: three files

| File | Content |
|---|---|
| `mask.png` | RGB, one pixel per cell, row 0 = north. `(0,0,0)` land, `(255,255,255)` water, `(0,255,0)` river inflow, `(0,0,255)` sea outlet, `(255,0,0)` Thanneermukkom barrage |
| `seed.png` | 8-bit grey: fraction of each cell covered by mats (0 to 255) |
| `wind.json` | hourly wind toward east/north (m/s) from `start` (the satellite pass), `forecast_start`, `forecast_hours`, `dx_m`, `barrage_open` |

`wind.json` starts at the satellite acquisition, so the sim spins up from the observed mats to
"now" with analysed wind before the 72 h forecast window begins. The pipeline also writes
`labels.npz/json` (OSM canal and place names for captions), `grid.json` and `meta.json`; the
simulation never reads them.

## Model

* **Water**: D2Q9 lattice-Boltzmann, BGK collision (tau 0.6), pull streaming and collision in one
  pass, half-way bounce-back walls. Wind is a body force balanced by linear bottom friction, so
  open water settles at about 2 % of wind speed. River inflows and the sea outlet (north edge,
  towards Kochi) are pressure boundaries with zero-gradient velocity. The barrage is a wall when
  closed and water when open. Velocities are clamped for stability.
* **Mats**: particles moving with the bilinear-sampled current plus 3 % of the wind (windage),
  slowed by crowding, pulled slightly together (rafts), with a small random walk. A mat that hits
  a bank sticks and breaks free only rarely. Mats reaching the sea outlet are flushed.
* **Jam**: dwell time per cell is the heatmap. A cell counts as *jammed* once mats have piled up
  at blocking density for 2 h within 120 m of a bank, bend, inlet or the barrage; the time that
  started is its ETA. Rafts crossing open water show in the heatmap but are not jams.
* **Units**: 40 m cells, time step sized so 0.5 m/s maps to lattice speed 0.08 (about 6.4 s).

`deno task selftest` checks the solver on the GPU against known answers: rest state, mass
conservation, stability in strong wind, wind-driven channel current (2 % of wind), mat drift
(current + 3 % windage), closed/open barrage, pile-up against a bank, river-driven flow.

Caveat: this is incompressible 2D LBM, not shallow-water equations, and nothing is calibrated
against observed drift. It is a drift likelihood, not hydrology.

## Satellite seeding

Sentinel-2 L2A from the Microsoft Planetary Computer STAC API. Mats are detected at the native
10 m: NDVI > 0.45 inside the OSM water mask eroded by one pixel off every bank (so palms and paddy
mixed into bank pixels are not counted), using the scene classification layer to drop cloud and
shadow, and correcting the +1000 offset that baseline 04.00+ products add. Each 40 m cell gets
the fraction of its area covered. The newest scene with at least 70 % of the water cloud-free is
used. Canals narrower than about 20 m cannot be observed at 10 m.

## Lake health (satellite record since Jan 2025)

`python -m hyacinth health` adds every new clear Sentinel-2 pass to `data/health/` (resumable;
the first run backfills from 1 Jan 2025, about 30 s per date) and `python -m hyacinth
health-publish` writes `web/public/data/health/`: `health.json`, `hotspots.png`, `latest.png`.
`daily` runs both. Per pass and per zone (OSM places with enough open water), at 20 m on water at
least 40 m from any bank (narrow canals mix with bank vegetation and are left out):

| Indicator | How | Meaning |
|---|---|---|
| Hyacinth cover | NDVI > 0.45 | % of cloud-free water under floating mats |
| Possible debris | Floating Debris Index (Biermann et al. 2020) above the pass's own open-water level, NDVI near 0, 40 m+ from mats and banks, 0.08-1 ha patches | low confidence per pass; recurring spots matter |
| Algae | NDCI over open water | chlorophyll-a proxy |
| Turbidity | NDTI over open water | sediment / mud proxy |
| Health index | mean percentile rank of the four, against the zone's own record | 100 = best this zone has been; relative, not a standard |

Daily **geostationary** data (EUMETSAT Meteosat IODC / JMA Himawari, 15-minute surface solar
radiation via the Open-Meteo satellite radiation API) gives cloudiness (measured vs a clear-sky
model) and sunshine every day, and a "now" reading minutes old. It fills the days between
Sentinel-2 passes and shows why the monsoon leaves gaps. Rain and temperature (ERA5 and forecast
models) are context, labelled as model data. Satellites cannot give live debris positions; the
newest detections are always from the last clear pass, and the site says how old that is.

## Ask Hyacinth (AI on the site)

A chat panel on the site, in English or Malayalam. `web/server/ask.ts` makes one OpenAI Chat
Completions call per request (model from `OPENAI_MODEL`) with a fixed system prompt, the day's
`facts.json`, and two tools that run **in the browser**, against the live GPU simulation:

* `run_scenario(barrage, wind, hours)`: restores the forecast-start checkpoint, runs today's
  conditions once (cached) and the what-if as fast as the GPU allows while the user watches, reads
  the jam/ETA rasters back and sums them per OSM place (`places.png` + `places.json`, written by
  `publish`). The comparison goes back to the model, which answers from it.
* `show_place(place)`: flies the 3D camera there and drops a pin.

With the lake-health dataset published it also gets `lake_health`, `metric_series`,
`rank_zones`, `debris_sightings`, `sky_now` and `show_map_layer`, all answered in the browser
from `health.json`.

The browser holds the conversation and echoes assistant turns back unchanged; the server
validates it (alternating roles, user turns only text or tool results, size caps). Deployed as a
Cloudflare Pages Function (`web/functions/api/ask.ts`): set the `OPENAI_API_KEY` secret and `OPENAI_MODEL` variable on the
Pages project. In `npm run dev` the same handler is mounted by `vite.config.ts` and uses
`OPENAI_API_KEY` / `OPENAI_MODEL` from the repo-root `.env` (copy `.env.example`). Without a key the panel says so and everything else works.

## Performance

* The solver dispatches only the 8x8 tiles that contain water (about a third of the Vembanad grid),
  reads one packed neighbour-wall mask per cell instead of nine neighbour lookups, and encodes a
  whole batch of steps as one compute pass (per-step uniforms picked by dynamic offset, no copies).
  Mats are dispatched only over live particle slots.
* The page never queues more than two frames on the GPU and sizes the number of sim steps per frame
  from measured GPU time (`web/src/pacer.ts`), so drawing stays smooth and a slower GPU simply
  simulates fewer hours per second. The 3D sky is drawn after opaque geometry (only uncovered pixels).
* The daily render saves the mats at forecast start (`state.bin`); the browser loads it and only
  warms up the currents for 2 h of sim time instead of replaying the whole spin-up from the satellite
  pass. Reset and what-if runs restore a GPU-side checkpoint (buffer copies, no CPU round trip).
* `deno task bench data/latest` prints microseconds per step for a dataset.

## Captions

`aggregate.py` turns the jam rasters into facts in code: the top jam segments by OSM place and
canal name, newly blocked area, ETA, the weekday in English and Malayalam (IST), wind summary and
barrage state. `caption.py` sends only those facts to the OpenAI model in `OPENAI_MODEL` (JSON schema
output, told never to add or transliterate names) and appends the disclaimer itself. Without
credentials, or if the call fails, a template caption is used. Synthetic runs are always
labelled as tests.

## Delivery and scheduling

* Telegram Bot API: video + caption to a public channel; people forward it into WhatsApp groups
  (WhatsApp Channels have no posting API).
* Static site on Cloudflare Pages: today's forecast and the interactive sim
  (`?data=latest|hindcast|synthetic`).
* `.github/workflows/daily.yml`: daily at 06:00 IST on GitHub Actions with Mesa lavapipe
  (software Vulkan). Secrets: `OPENAI_API_KEY` (and variable `OPENAI_MODEL`), `TELEGRAM_BOT_TOKEN`, `CLOUDFLARE_API_TOKEN`,
  `CLOUDFLARE_ACCOUNT_ID`; variables: `TELEGRAM_CHAT_ID`, optionally `HYACINTH_BARRAGE=open|closed`.
  A cron job on any machine with a GPU works the same way: `python -m hyacinth daily`.

## Configuration

`pipeline/hyacinth/config.py`: bounding box, grid size, barrage line and seasonal default
(closed mid-December to March unless `HYACINTH_BARRAGE` says otherwise), canal widths, detection
thresholds. Physics knobs are `DEFAULT_PARAMS` in `sim/engine.ts`.

## Known limits

* Uncalibrated: pitch it as likelihood, not a guarantee.
* Monsoon cloud blinds Sentinel-2 for weeks; Sentinel-1 radar would be the fallback (not built).
* OSM canals are messy. Narrow canals are widened to 2 cells (80 m) so the solver can carry flow.
* Canals run around the Thanneermukkom bund on both banks, so closing the barrage closes the lake
  but not every canal link; Kuttanad's many smaller regulators are not modelled.
* Spatially uniform wind from one Open-Meteo point.

Data: Copernicus Sentinel-2 (ESA), © OpenStreetMap contributors (ODbL), Open-Meteo (CC BY 4.0).
