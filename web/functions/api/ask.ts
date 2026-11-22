// Cloudflare Pages Function: POST /api/ask. Set OPENAI_API_KEY (secret) and
// OPENAI_MODEL on the Pages project (Settings > Variables and Secrets).

import { handleAsk } from "../../server/ask.ts";

interface Env {
  OPENAI_API_KEY?: string;
  OPENAI_MODEL?: string;
  ASSETS: { fetch: (req: Request | string) => Promise<Response> };
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", "cache-control": "no-store" } });

export const onRequestPost = async (ctx: { request: Request; env: Env }): Promise<Response> => {
  const { request, env } = ctx;
  if (!env.OPENAI_API_KEY || !env.OPENAI_MODEL) return json({ error: "The assistant is not configured on this site." }, 503);
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return json({ error: "body must be JSON" }, 400);
  }
  try {
    const result = await handleAsk(body, {
      apiKey: env.OPENAI_API_KEY,
      model: env.OPENAI_MODEL,
      loadFacts: async (name) => {
        const r = await env.ASSETS.fetch(new URL(`/data/${name}/facts.json`, request.url).toString());
        return r.ok ? r.json() : null;
      },
      loadHealth: async () => {
        const r = await env.ASSETS.fetch(new URL("/data/health/health.json", request.url).toString());
        return r.ok ? r.json() : null;
      },
    });
    return result.ok ? json(result) : json({ error: result.error }, result.status);
  } catch (e) {
    console.error("ask:", e);
    return json({ error: "Something went wrong." }, 500);
  }
};
