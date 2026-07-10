export interface Env {
  OPENAI_API_KEY: string;
  VISION_API_KEY: string;
  PROXY_TOKEN: string;
}

const OPENAI_BASE = "https://api.openai.com/v1";
const VISION_BASE = "https://generativelanguage.googleapis.com";

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

    const response = await fetch(upstreamUrl.toString(), {
      method: req.method,
      headers: filterHeaders(req, apiKey, authMode),
      body: req.method === "GET" || req.method === "HEAD" ? undefined : req.body,
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

      if (url.pathname.startsWith("/openai")) {
        return forward(request, OPENAI_BASE, env.OPENAI_API_KEY, "/openai", "bearer");
      }

      if (url.pathname.startsWith("/vision")) {
        return forward(request, VISION_BASE, env.VISION_API_KEY, "/vision", "google_api_key");
      }

      return new Response("ok", { status: 200 });
    },
  };
