export interface Env {
  FIREWORKS_API_KEY: string;
  OPENROUTER_API_KEY: string;
  PROXY_TOKEN: string;
  FIREWORKS_MODEL: string;
  OPENROUTER_MODEL: string;
}

const FIREWORKS_BASE = "https://api.fireworks.ai/inference/v1";
const OPENROUTER_BASE = "https://openrouter.ai/api/v1";

function unauthorized() {
  return new Response("unauthorized", { status: 401 });
}

function filterHeaders(req: Request, apiKey: string) {
  const headers = new Headers(req.headers);
  headers.delete("authorization");
  headers.delete("x-goog-api-key");
  headers.set("Authorization", `Bearer ${apiKey}`);
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

async function buildForwardBody(req: Request, model?: string) {
  if (req.method === "GET" || req.method === "HEAD") {
    return undefined;
  }
  const contentType = req.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) {
    return req.body;
  }
  const payload = await req.json<any>();
  if (model && payload && typeof payload === "object" && !Array.isArray(payload)) {
    payload.model = model;
  }
  return JSON.stringify(payload);
}

async function forward(
  req: Request,
  targetBase: string,
  apiKey: string,
  pathPrefix: string,
  model?: string,
) {
  const url = new URL(req.url);
  const upstreamPath = url.pathname.replace(pathPrefix, "") || "/";
  const upstreamUrl = buildUpstreamUrl(targetBase, upstreamPath, url.search);

  const response = await fetch(upstreamUrl.toString(), {
    method: req.method,
    headers: filterHeaders(req, apiKey),
    body: await buildForwardBody(req, model),
    redirect: "follow",
  });
  return rewriteUploadUrlHeader(response, req, pathPrefix);
}

function rewriteUploadUrlHeader(response: Response, req: Request, pathPrefix: string) {
  const upstreamUploadUrl = response.headers.get("x-goog-upload-url");
  if (!upstreamUploadUrl) {
    return response;
  }

  const requestUrl = new URL(req.url);
  const upstreamUrl = new URL(upstreamUploadUrl);
  upstreamUrl.protocol = requestUrl.protocol;
  upstreamUrl.host = requestUrl.host;
  upstreamUrl.pathname = `${pathPrefix}${upstreamUrl.pathname}`;

  const headers = new Headers(response.headers);
  headers.set("x-goog-upload-url", upstreamUrl.toString());

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.headers.get("x-proxy-token") !== env.PROXY_TOKEN) {
      return unauthorized();
    }

    const url = new URL(request.url);

    if (url.pathname.startsWith("/vision")) {
      return forward(request, FIREWORKS_BASE, env.FIREWORKS_API_KEY, "/vision", env.FIREWORKS_MODEL);
    }

    if (url.pathname.startsWith("/caption")) {
      return forward(request, FIREWORKS_BASE, env.FIREWORKS_API_KEY, "/caption", env.FIREWORKS_MODEL);
    }

    if (url.pathname.startsWith("/judge")) {
      return forward(request, FIREWORKS_BASE, env.FIREWORKS_API_KEY, "/judge", env.FIREWORKS_MODEL);
    }

    if (url.pathname.startsWith("/openrouter/vision")) {
      return forward(request, OPENROUTER_BASE, env.OPENROUTER_API_KEY, "/openrouter/vision", env.OPENROUTER_MODEL);
    }

    if (url.pathname.startsWith("/openrouter/caption")) {
      return forward(request, OPENROUTER_BASE, env.OPENROUTER_API_KEY, "/openrouter/caption", env.OPENROUTER_MODEL);
    }

    if (url.pathname.startsWith("/openrouter/judge")) {
      return forward(request, OPENROUTER_BASE, env.OPENROUTER_API_KEY, "/openrouter/judge", env.OPENROUTER_MODEL);
    }

    return new Response("ok", { status: 200 });
  },
};
