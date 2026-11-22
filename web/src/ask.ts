// "Ask Hyacinth" client: a short conversation with the forecast assistant.
// The server (server/ask.ts) makes one model call per request; tool calls it
// returns run here, against the live GPU simulation, and their results go
// back on the next request. Assistant turns are echoed back unchanged.

import type OpenAI from "openai";

type Message = OpenAI.Chat.ChatCompletionMessageParam;
type ToolCall = OpenAI.Chat.ChatCompletionMessageToolCall;

export interface ScenarioInput {
  barrage: "as_forecast" | "open" | "closed";
  wind: "forecast" | "fixed";
  wind_from_deg: number;
  wind_speed_ms: number;
  hours: 24 | 48 | 72;
}

export interface AskTools {
  /** Needs the live GPU simulation; absent without WebGPU. */
  runScenario?(input: ScenarioInput, progress: (fraction: number, label: string) => void): Promise<unknown>;
  showPlace?(place: string): unknown;
  /** Data tools (lake-health dataset), by tool name. */
  extra?: Record<string, (args: Record<string, unknown>) => unknown>;
}

const TOOL_LABELS: Record<string, string> = {
  lake_health: "Checked lake health",
  metric_series: "Read the satellite record",
  rank_zones: "Ranked zones",
  debris_sightings: "Looked up debris sightings",
  sky_now: "Read geostationary satellite conditions",
  show_map_layer: "Switched the map",
};

const MAX_ROUNDS = 8;
const MALAYALAM = /[ഀ-ൿ]/;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

export function initAsk(dataset: string, tools: AskTools): void {
  const log = $<HTMLDivElement>("ask-log");
  const form = $<HTMLFormElement>("ask-form");
  const input = $<HTMLTextAreaElement>("ask-input");
  const send = $<HTMLButtonElement>("ask-send");
  const presets = [...document.querySelectorAll<HTMLButtonElement>("[data-ask]")];
  const messages: Message[] = [];
  let busy = false;
  let disabled = false;

  const setBusy = (b: boolean) => {
    busy = b;
    send.disabled = b || disabled;
    for (const p of presets) p.disabled = b || disabled;
    log.setAttribute("aria-busy", String(b));
  };

  const scroll = () => log.scrollTo({ top: log.scrollHeight, behavior: "smooth" });

  const bubble = (role: "user" | "assistant" | "status" | "error", text = "") => {
    log.querySelector(".ask-empty")?.remove();
    const el = document.createElement("div");
    el.className = `msg ${role}`;
    if (text) renderText(el, text);
    log.append(el);
    scroll();
    return el;
  };

  const ask = async (question: string) => {
    question = question.trim();
    if (!question || busy || disabled) return;
    setBusy(true);
    bubble("user", question);
    const startLen = messages.length;
    messages.push({ role: "user", content: question });
    const thinking = bubble("status", "Thinking…");
    try {
      for (let round = 0; round < MAX_ROUNDS; round++) {
        const res = await fetch("./api/ask", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ dataset, messages }),
        });
        const body = (await res.json().catch(() => ({}))) as {
          message?: OpenAI.Chat.ChatCompletionMessage;
          finish_reason?: string;
          error?: string;
        };
        if (!res.ok || !body.message) {
          if (res.status === 503) {
            disabled = true;
            $("ask-note").hidden = false;
          }
          throw new Error(body.error ?? `HTTP ${res.status}`);
        }
        const msg = body.message;
        const calls = (msg.tool_calls ?? []).filter((c) => c.type === "function");
        // Echo the assistant turn back as-is (content + tool calls).
        messages.push({ role: "assistant", content: msg.content, ...(calls.length ? { tool_calls: calls } : {}) });
        if (msg.content?.trim()) bubble("assistant", msg.content.trim());
        if (calls.length === 0) break;

        for (const call of calls) {
          let content: string;
          try {
            content = JSON.stringify(await runTool(call));
          } catch (e) {
            content = `Error: ${(e as Error).message}`;
          }
          messages.push({ role: "tool", tool_call_id: call.id, content });
        }
      }
    } catch (e) {
      // Drop the failed exchange so the conversation stays valid.
      messages.length = startLen;
      bubble("error", `Couldn't get an answer: ${(e as Error).message}`);
    } finally {
      thinking.remove();
      setBusy(false);
    }
  };

  const runTool = async (call: ToolCall): Promise<unknown> => {
    if (call.type !== "function") throw new Error("unsupported tool type");
    let args: Record<string, unknown>;
    try {
      args = JSON.parse(call.function.arguments || "{}");
    } catch {
      throw new Error("tool arguments were not valid JSON");
    }
    const name = call.function.name;
    const data = tools.extra?.[name];
    if (data) {
      const out = data(args);
      const note = bubble("status");
      note.classList.add("done");
      note.textContent = TOOL_LABELS[name] ?? name;
      return out;
    }
    if (name === "show_place") {
      if (!tools.showPlace) throw new Error("the map is not available in this browser");
      return tools.showPlace(String(args.place ?? ""));
    }
    if (name === "run_scenario") {
      if (!tools.runScenario) throw new Error("what-if runs need the live simulation, which needs WebGPU in this browser");
      const input = parseScenario(args);
      const status = bubble("status");
      const label = document.createElement("span");
      const bar = document.createElement("i");
      bar.className = "bar";
      status.append(label, bar);
      try {
        return await tools.runScenario!(input, (f, text) => {
          label.textContent = `${text} · ${Math.round(f * 100)}%`;
          bar.style.setProperty("--p", `${(f * 100).toFixed(1)}%`);
        });
      } finally {
        status.classList.add("done");
        label.textContent = `Ran on your GPU: ${describeScenario(input)}`;
        bar.remove();
      }
    }
    throw new Error(`unknown tool ${call.function.name}`);
  };

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value;
    input.value = "";
    ask(q);
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });
  for (const p of presets) p.addEventListener("click", () => ask(p.dataset.ask ?? p.textContent ?? ""));
  setBusy(false);
}

function parseScenario(a: Record<string, unknown>): ScenarioInput {
  const barrage = a.barrage === "open" || a.barrage === "closed" ? a.barrage : "as_forecast";
  const wind = a.wind === "fixed" ? "fixed" : "forecast";
  const hours = a.hours === 24 || a.hours === 48 ? a.hours : 72;
  const deg = Number(a.wind_from_deg);
  const speed = Number(a.wind_speed_ms);
  return {
    barrage,
    wind,
    wind_from_deg: Number.isFinite(deg) ? ((deg % 360) + 360) % 360 : 270,
    wind_speed_ms: Number.isFinite(speed) ? Math.min(15, Math.max(0, speed)) : 4,
    hours,
  };
}

const COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];

export function describeScenario(s: ScenarioInput): string {
  const parts = [`${s.hours} h`];
  if (s.barrage !== "as_forecast") parts.push(`barrage ${s.barrage}`);
  if (s.wind === "fixed") parts.push(`wind ${s.wind_speed_ms} m/s from ${COMPASS[Math.round(s.wind_from_deg / 45) % 8]}`);
  if (parts.length === 1) parts.push("today's conditions");
  return parts.join(", ");
}

/** Plain text with "- " list lines and **bold**, built as DOM (never innerHTML). */
function renderText(el: HTMLElement, text: string) {
  if (MALAYALAM.test(text)) el.classList.add("ml");
  let list: HTMLUListElement | null = null;
  for (const line of text.split("\n")) {
    const t = line.trim();
    if (!t) {
      list = null;
      continue;
    }
    const item = /^[-*•]\s+/.test(t) || /^\d+[.)]\s+/.test(t);
    let target: HTMLElement;
    if (item) {
      if (!list) el.append((list = document.createElement("ul")));
      target = document.createElement("li");
      list.append(target);
    } else {
      list = null;
      target = document.createElement("p");
      el.append(target);
    }
    const body = item ? t.replace(/^([-*•]|\d+[.)])\s+/, "") : t;
    body.split(/(\*\*[^*]+\*\*)/).forEach((part) => {
      if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
        const b = document.createElement("strong");
        b.textContent = part.slice(2, -2);
        target.append(b);
      } else if (part) target.append(part);
    });
  }
}
