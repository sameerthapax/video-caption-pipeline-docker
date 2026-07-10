  export interface Env {
    FIREWORKS_API_KEY: string;
    OPENAI_API_KEY: string;
    GOOGLE_GEMINI_API_KEY: string;
    PROXY_TOKEN: string;
  }

  const FIREWORKS_BASE = "https://api.fireworks.ai/inference/v1";
  const OPENAI_BASE = "https://api.openai.com/v1";
  const GOOGLE_GEMINI_BASE = "https://generativelanguage.googleapis.com";

  function unauthorized() {
    return new Response("unauthorized", { status: 401 });
  }

  type AuthMode = "bearer" | "google_api_key";

  function filterHeaders(req: Request, apiKey: string, authMode: AuthMode) {
    const headers = new Headers(req.headers);
    headers.delete("authorization");
    headers.delete("x-goog-api-key");
    if (authMode === "google_api_key") {
      headers.set("x-goog-api-key", apiKey);
    } else {
      headers.set("Authorization", `Bearer ${apiKey}`);
    }
    headers.set("Content-Type", "application/json");
    headers.delete("host");
    headers.delete("x-proxy-token");
    return headers;
  }

  function buildUpstreamUrl(targetBase: string, upstreamPath: string, search: string) {
    const normalizedBase = targetBase.endsWith("/") ? targetBase : `${targetBase}/`;
    const normalizedPath = upstreamPath.replace(/^\/+/, "");
    return new URL(`${normalizedPath}${search}`, normalizedBase);
  }

  async function forward(
    req: Request,
    targetBase: string,
    apiKey: string,
    pathPrefix: string,
    authMode: AuthMode = "bearer",
  ) {
    const url = new URL(req.url);
    const upstreamPath = url.pathname.replace(pathPrefix, "") || "/";
    const upstreamUrl = buildUpstreamUrl(targetBase, upstreamPath, url.search);

    return fetch(upstreamUrl.toString(), {
      method: req.method,
      headers: filterHeaders(req, apiKey, authMode),
      body: req.method === "GET" || req.method === "HEAD" ? undefined : req.body,
      redirect: "follow",
    });
  }

  export default {
    async fetch(request: Request, env: Env): Promise<Response> {
      if (request.headers.get("x-proxy-token") !== env.PROXY_TOKEN) {
        return unauthorized();
      }

      const url = new URL(request.url);

      if (url.pathname.startsWith("/fireworks")) {
        return forward(request, FIREWORKS_BASE, env.FIREWORKS_API_KEY, "/fireworks", "bearer");
      }

      if (url.pathname.startsWith("/openai")) {
        return forward(request, OPENAI_BASE, env.OPENAI_API_KEY, "/openai", "bearer");
      }

      if (url.pathname.startsWith("/google")) {
        return forward(request, GOOGLE_GEMINI_BASE, env.GOOGLE_GEMINI_API_KEY, "/google", "google_api_key");
      }

      return new Response("ok", { status: 200 });
    },
  };
