import {
	env,
	createExecutionContext,
	waitOnExecutionContext,
} from "cloudflare:test";
import { describe, it, expect, vi, afterEach } from "vitest";
import worker from "../src/index";

// For now, you'll need to do something like this to get a correctly-typed
// `Request` to pass to `worker.fetch()`.
const IncomingRequest = Request<unknown, IncomingRequestCfProperties>;

describe("caption proxy worker", () => {
	afterEach(() => {
		vi.restoreAllMocks();
	});

	it("rejects requests without the proxy token", async () => {
		const request = new IncomingRequest("http://example.com");
		const ctx = createExecutionContext();
		const response = await worker.fetch(request, env, ctx);
		await waitOnExecutionContext(ctx);
		expect(response.status).toBe(401);
		expect(await response.text()).toBe("unauthorized");
	});

	it("forwards caption traffic with injected model and bearer auth", async () => {
		const upstreamResponse = new Response(JSON.stringify({ ok: true }), {
			status: 200,
			headers: { "content-type": "application/json" },
		});
		const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(upstreamResponse);

		const request = new IncomingRequest("http://example.com/caption/chat/completions?foo=bar", {
			method: "POST",
			headers: {
				"x-proxy-token": env.PROXY_TOKEN,
				"authorization": "Bearer should-be-replaced",
				"content-type": "application/json",
			},
			body: JSON.stringify({ messages: [] }),
		});
		const ctx = createExecutionContext();
		const response = await worker.fetch(request, env, ctx);
		await waitOnExecutionContext(ctx);

		expect(fetchMock).toHaveBeenCalledTimes(1);
		const [url, init] = fetchMock.mock.calls[0];
		expect(url).toBe("https://api.fireworks.ai/inference/v1/chat/completions?foo=bar");
		expect((init?.headers as Headers).get("Authorization")).toBe(`Bearer ${env.FIREWORKS_API_KEY}`);
		expect((init?.headers as Headers).get("x-proxy-token")).toBeNull();
		expect(init?.body).toBe(JSON.stringify({ messages: [], model: env.FIREWORKS_MODEL }));
		expect(response.status).toBe(200);
		expect(await response.text()).toBe(JSON.stringify({ ok: true }));
	});
});
