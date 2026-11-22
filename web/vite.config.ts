import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { defineConfig, loadEnv, type Plugin } from "vite";

/** Serve POST /api/ask in `npm run dev` with the same handler as the Pages Function. */
function askApi(env: Record<string, string>): Plugin {
  return {
    name: "hyacinth-ask-api",
    configureServer(server) {
      server.middlewares.use("/api/ask", async (req, res) => {
        const send = (status: number, body: unknown) => {
          res.statusCode = status;
          res.setHeader("content-type", "application/json");
          res.end(JSON.stringify(body));
        };
        if (req.method !== "POST") return send(405, { error: "POST only" });
        const apiKey = env.OPENAI_API_KEY;
        const model = env.OPENAI_MODEL;
        if (!apiKey || !model) {
          return send(503, { error: "The assistant is not configured: set OPENAI_API_KEY and OPENAI_MODEL in the repo's .env (see .env.example)." });
        }
        let raw = "";
        for await (const chunk of req) raw += chunk;
        try {
          const { handleAsk } = (await server.ssrLoadModule("/server/ask.ts")) as typeof import("./server/ask.ts");
          const result = await handleAsk(JSON.parse(raw), {
            apiKey,
            model,
            loadFacts: async (name) => {
              try {
                return JSON.parse(await readFile(new URL(`./public/data/${name}/facts.json`, import.meta.url), "utf8"));
              } catch {
                return null;
              }
            },
            loadHealth: async () => {
              try {
                return JSON.parse(await readFile(new URL("./public/data/health/health.json", import.meta.url), "utf8"));
              } catch {
                return null;
              }
            },
          });
          if (result.ok) send(200, result);
          else send(result.status, { error: result.error });
        } catch (e) {
          console.error("[ask]", e);
          send(500, { error: "Something went wrong." });
        }
      });
    },
  };
}

// The simulation lives in ../sim and is shared with the Deno renderer.
// Secrets come from ../.env (repo root, shared with the Python pipeline). They
// are only read here on the dev server; nothing without a VITE_ prefix reaches
// the browser bundle.
export default defineConfig(({ mode }) => ({
  base: "./",
  plugins: [askApi(loadEnv(mode, fileURLToPath(new URL("..", import.meta.url)), ""))],
  resolve: {
    alias: {
      // ../sim/data.ts imports "fast-png"; resolve it from this package.
      "fast-png": fileURLToPath(new URL("./node_modules/fast-png", import.meta.url)),
    },
  },
  server: {
    fs: { allow: [".."] },
  },
  build: {
    target: "es2022",
  },
}));
