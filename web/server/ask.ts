// "Ask Hyacinth": a forecast assistant grounded in today's facts.json, with two
// tools that run IN THE BROWSER (the GPU simulation lives there):
//
//   run_scenario  re-run the forecast window with a different barrage state
//                 or wind, and return which places jam, how much, and when
//   show_place    point the map at a place named in the forecast
//
// This module is one stateless turn: the browser sends the conversation so
// far (assistant turns echoed back unchanged, tool results appended), we make
// one OpenAI Chat Completions call and return the assistant message. The
// browser runs any tool calls and calls again. The system prompt, tools and
// facts are fixed here on the server; the client cannot change them.
//
// Shared by the Cloudflare Pages Function (functions/api/ask.ts) and the Vite
// dev server (vite.config.ts). Configure with OPENAI_API_KEY and OPENAI_MODEL.

import OpenAI from "openai";

const MAX_MESSAGES = 40;
const MAX_BODY_CHARS = 120_000;
const MAX_QUESTION_CHARS = 1_000;

type Message = OpenAI.Chat.ChatCompletionMessageParam;

export interface AskRequest {
  dataset: string;
  messages: Message[];
}

export type AskResult =
  | { ok: true; message: OpenAI.Chat.ChatCompletionMessage; finish_reason: string }
  | { ok: false; status: number; error: string };

export interface AskEnv {
  apiKey: string;
  model: string;
  /** facts.json for a dataset, or null if it has none. */
  loadFacts: (dataset: string) => Promise<unknown | null>;
  /** health.json (lake-health dataset), or null if not published. */
  loadHealth: () => Promise<HealthMeta | null>;
}

interface HealthMeta {
  period: [string, string];
  zones: { name: string; name_ml: string | null }[];
  passes: { date: string[] };
  latest_pass: string | null;
}

const fn = (name: string, description: string, properties: Record<string, unknown>): OpenAI.Chat.ChatCompletionTool => ({
  type: "function",
  function: {
    name,
    description,
    strict: true,
    parameters: { type: "object", properties, required: Object.keys(properties), additionalProperties: false },
  },
});

const ZONE = { type: "string", description: 'Zone name exactly as listed in the dataset, or "Whole lake".' };
const METRIC = {
  type: "string",
  enum: ["health", "hyacinth_pct", "debris_ha_per_km2", "chlorophyll", "turbidity"],
  description: "health = relative health index 0-100 (higher better); the others: higher is worse.",
};
const DATE = { type: "string", description: 'YYYY-MM-DD, or "latest".' };

const HEALTH_TOOLS: OpenAI.Chat.ChatCompletionTool[] = [
  fn("lake_health", "Health index and its four indicators for a zone at the satellite pass nearest a date, compared with that zone's usual level for the month and its previous pass.", { zone: ZONE, date: DATE }),
  fn("metric_series", "Time series of one indicator for a zone between two dates, per satellite pass or as monthly medians. Use for trends, seasons and comparisons.", {
    metric: METRIC,
    zone: ZONE,
    start: { type: "string", description: "YYYY-MM-DD" },
    end: { type: "string", description: 'YYYY-MM-DD or "latest"' },
    by: { type: "string", enum: ["pass", "month"] },
  }),
  fn("rank_zones", "Rank zones by an indicator: averaged over the whole record (period=all) or at the latest clear pass (period=latest). Use for hotspots and 'where is it worst'.", {
    metric: METRIC,
    period: { type: "string", enum: ["all", "latest"] },
    top: { type: "integer", description: "1-20" },
  }),
  fn("debris_sightings", "Possible floating-debris patches detected by Sentinel-2 between two dates, optionally in one zone, with location, size and how often that spot recurs.", {
    start: { type: "string", description: "YYYY-MM-DD" },
    end: { type: "string", description: 'YYYY-MM-DD or "latest"' },
    zone: ZONE,
  }),
  fn("sky_now", "Current conditions from geostationary satellites (sunlight now, cloudiness today), the last 7 days of cloudiness/sunshine/rain, and how old the last clear Sentinel-2 pass is.", {}),
  fn("show_map_layer", "Switch the map: hotspots = how often mats/possible debris appeared since Jan 2025; latest = the most recent clear satellite pass; forecast = the live 72 h simulation.", {
    layer: { type: "string", enum: ["hotspots", "latest", "forecast"] },
  }),
];

const TOOLS: OpenAI.Chat.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "run_scenario",
      description:
        "Re-run the forecast on the user's GPU from the forecast start with a different Thanneermukkom barrage " +
        "state and/or a fixed wind, and compare it with an unchanged baseline run (also computed live, so " +
        "both use the same method). Returns, for each named place, the jammed area in hectares and the hours " +
        "after the forecast start when new jams form, plus totals. Takes roughly 10-60 seconds; the user " +
        "watches it run. Use it for any 'what if' question about the barrage or the wind, never guess those.",
      strict: true,
      parameters: {
        type: "object",
        properties: {
          barrage: {
            type: "string",
            enum: ["as_forecast", "open", "closed"],
            description: "Barrage state for the run. as_forecast keeps today's state.",
          },
          wind: {
            type: "string",
            enum: ["forecast", "fixed"],
            description: "forecast = today's hourly wind forecast; fixed = constant wind from wind_from_deg at wind_speed_ms.",
          },
          wind_from_deg: {
            type: "number",
            description: "Direction the wind blows FROM, meteorological degrees (270 = from the west). Ignored unless wind is fixed.",
          },
          wind_speed_ms: {
            type: "number",
            description: "Wind speed in m/s (0-15). Ignored unless wind is fixed.",
          },
          hours: {
            type: "integer",
            enum: [24, 48, 72],
            description: "How many hours after the forecast start to simulate. Use the shortest that answers the question.",
          },
        },
        required: ["barrage", "wind", "wind_from_deg", "wind_speed_ms", "hours"],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "show_place",
      description:
        "Point the user's map at a place, by its exact name as it appears in the forecast facts or a scenario " +
        "result. Call it when your answer is mainly about one place.",
      strict: true,
      parameters: {
        type: "object",
        properties: { place: { type: "string", description: "Exact place name from the data." } },
        required: ["place"],
        additionalProperties: false,
      },
    },
  },
];

const HEALTH_PROMPT = (h: HealthMeta) => `

Lake-health dataset (tools: lake_health, metric_series, rank_zones, debris_sightings, sky_now, show_map_layer):
- ${h.passes.date.length} clear Sentinel-2 passes from ${h.period[0]} to ${h.passes.date.at(-1) ?? "-"} (latest clear pass ${h.latest_pass ?? "-"}), analysed at 20 m per zone; daily cloudiness and sunshine from geostationary satellites (Meteosat IODC / Himawari) every 15 min; rain and temperature are model data.
- Indicators: hyacinth cover (% of clear water), possible floating debris (ha per km2, LOW confidence per pass: small young hyacinth clumps can look similar; recurring spots are more reliable), chlorophyll proxy (NDCI, algae), turbidity proxy (NDTI). All relative, from surface reflectance, not lab measurements.
- Health index 0-100 ranks a zone against its own record (100 = best seen); it is not a water-quality standard. Say so when you quote it.
- Satellites never give live debris positions: the newest detections are from the last clear pass; say how old it is.
- Use the tools for any question about trends, seasons, pollution, algae, mud, debris, "lake health" or "where is it worst"; quote dates and numbers from them. Call show_map_layer when a map helps.
- Zones: ${h.zones.map((z) => z.name).join("; ")}.`;

const SYSTEM = `You are Hyacinth (കുളവാഴ), the assistant on a free public website that forecasts where water hyacinth will drift and jam on Vembanad lake and the Kuttanad canals in Kerala over the next 72 hours.

Who asks you: boat and ferry operators and houseboat crews deciding which routes to use; panchayat and clearance crews deciding where to send people and harvesters before mats lock in; farmers and irrigation staff thinking about the Thanneermukkom barrage; residents.

How to answer:
- Ground every claim in the forecast facts below or in a run_scenario result. Never invent place names, times or areas, and never transliterate or translate a place name yourself: use the "place" and "place_ml" strings exactly as given.
- A place that does not appear in the data simply has no jam forecast in the window. Say that, but do not promise it will be clear.
- For "what if" questions about the barrage or the wind, call run_scenario and answer from the comparison it returns. Never guess the effect.
- For crew planning, rank places by how soon new jams form (eta) and how much new jam area there is; earlier and larger first.
- Times: give weekday and time in IST (the facts give by_time_ist; for scenario results add eta hours to forecast_start_ist).
- This is an uncalibrated simulation: say "likely" rather than "will", and if the dataset is synthetic, say clearly it is a test, not a real forecast.
- Reply in the language the user wrote in (Malayalam or English). Keep it short: at most about 120 words, plain sentences or a short list, no headings, no tables.
- Only help with questions about the lake, canals, hyacinth, water quality, this forecast, the lake-health data and how to use the site. Politely decline anything else.`;

/** Light checks: shape, size, and only the roles a browser conversation can produce. */
function validate(body: unknown): AskRequest | string {
  if (!body || typeof body !== "object") return "body must be JSON";
  const { dataset, messages } = body as Partial<AskRequest>;
  if (typeof dataset !== "string" || !/^[a-z0-9_-]{1,32}$/.test(dataset)) return "bad dataset";
  if (!Array.isArray(messages) || messages.length === 0 || messages.length > MAX_MESSAGES) return "bad messages";
  if (JSON.stringify(messages).length > MAX_BODY_CHARS) return "conversation too long; start a new one";
  if (messages[0]?.role !== "user") return "conversation must start with the user";
  for (const m of messages) {
    if (!m || typeof m !== "object") return "bad message";
    if (m.role === "user") {
      if (typeof m.content !== "string" || m.content.length > MAX_QUESTION_CHARS) return "questions must be text under 1000 characters";
    } else if (m.role === "tool") {
      if (typeof m.tool_call_id !== "string" || typeof m.content !== "string") return "bad tool result";
    } else if (m.role !== "assistant") {
      return "only user, assistant and tool messages are allowed";
    }
  }
  const last = messages[messages.length - 1].role;
  if (last !== "user" && last !== "tool") return "last message must be a question or tool results";
  return { dataset, messages };
}

export async function handleAsk(body: unknown, env: AskEnv): Promise<AskResult> {
  const req = validate(body);
  if (typeof req === "string") return { ok: false, status: 400, error: req };
  const [facts, health] = await Promise.all([env.loadFacts(req.dataset), env.loadHealth().catch(() => null)]);
  if (!facts) return { ok: false, status: 400, error: `no forecast facts for dataset "${req.dataset}"` };

  const client = new OpenAI({ apiKey: env.apiKey });
  try {
    const completion = await client.chat.completions.create({
      model: env.model,
      max_completion_tokens: 4000,
      // Luna-class models reject function tools on Chat Completions unless reasoning is off.
      reasoning_effort: "none",
      messages: [
        // Frozen per dataset and day, so the prefix is reusable across visitors.
        {
          role: "system",
          content: `${SYSTEM}\n\nForecast facts (dataset "${req.dataset}"):\n${JSON.stringify(facts)}${health ? HEALTH_PROMPT(health) : ""}`,
        },
        ...req.messages,
      ],
      tools: health ? [...TOOLS, ...HEALTH_TOOLS] : TOOLS,
    });
    const choice = completion.choices[0];
    if (!choice) return { ok: false, status: 502, error: "empty response" };
    const message = choice.message;
    if (message.refusal) {
      return {
        ok: true,
        finish_reason: "stop",
        message: { role: "assistant", content: "Sorry, I can't help with that. Ask me about the hyacinth forecast.", refusal: null },
      };
    }
    return { ok: true, message, finish_reason: choice.finish_reason };
  } catch (e) {
    if (e instanceof OpenAI.AuthenticationError || e instanceof OpenAI.PermissionDeniedError) {
      return { ok: false, status: 503, error: "The assistant is not configured on this site." };
    }
    if (e instanceof OpenAI.NotFoundError) return { ok: false, status: 503, error: `Model "${env.model}" was not found; check OPENAI_MODEL.` };
    if (e instanceof OpenAI.RateLimitError) return { ok: false, status: 429, error: "Busy right now, try again in a minute." };
    if (e instanceof OpenAI.BadRequestError) return { ok: false, status: 400, error: e.message };
    if (e instanceof OpenAI.APIError) return { ok: false, status: 502, error: `Upstream error ${e.status ?? ""}`.trim() };
    throw e;
  }
}
