import { afterEach, expect, it, vi } from "vitest";
import { ApiError, postAction } from "./api";

const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

afterEach(() => vi.unstubAllGlobals());

function fakeFetch(failures: number, status = 200) {
  const bodies: { action_id: string; text?: string }[] = [];
  const urls: string[] = [];
  vi.stubGlobal("fetch", async (url: string, init: RequestInit) => {
    urls.push(url);
    bodies.push(JSON.parse(String(init.body)));
    if (bodies.length <= failures) throw new TypeError("Failed to fetch");
    return new Response(
      JSON.stringify(
        status === 200
          ? { id: "run-1", state: "executing" }
          : { error: { code: "RUN_SETTLED", message: "This run is stopping or finished." } },
      ),
      { status },
    );
  });
  return { bodies, urls };
}

it("attaches an action_id and reuses it when the first response is lost", async () => {
  const { bodies, urls } = fakeFetch(1);
  const result = await postAction<{ state: string }>(
    "/remotea/runs/run-1/message",
    { text: "Also update the docs.", mode: "steer" },
    { delayMs: 0 },
  );
  expect(result.state).toBe("executing");
  expect(bodies).toHaveLength(2);
  expect(bodies[0].action_id).toMatch(uuid);
  expect(bodies[1]).toEqual(bodies[0]);
  expect(urls).toEqual([
    "/api/v1/remotea/runs/run-1/message",
    "/api/v1/remotea/runs/run-1/message",
  ]);
});

it("gives each call its own action_id", async () => {
  const { bodies } = fakeFetch(0);
  await postAction("/remotea/runs/run-1/stop", { mode: "interrupt" });
  await postAction("/remotea/runs/run-1/stop", { mode: "interrupt" });
  expect(bodies[0].action_id).not.toBe(bodies[1].action_id);
});

it("gives up after the retries and does not retry a refusal", async () => {
  const lost = fakeFetch(5);
  await expect(
    postAction("/remotea/runs/run-1/approve", {}, { retries: 2, delayMs: 0 }),
  ).rejects.toMatchObject({ code: "NETWORK" });
  expect(lost.bodies).toHaveLength(3);
  expect(new Set(lost.bodies.map((b) => b.action_id)).size).toBe(1);

  const refused = fakeFetch(0, 409);
  const error: unknown = await postAction("/remotea/runs/run-1/stop", {}, { delayMs: 0 }).catch((e) => e);
  expect(error).toBeInstanceOf(ApiError);
  expect((error as ApiError).code).toBe("RUN_SETTLED");
  expect(refused.bodies).toHaveLength(1);
});
