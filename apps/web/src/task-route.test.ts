import { expect, test } from "vitest";
import { routeNote, type TaskReadiness } from "./TaskRoute";

const base: TaskReadiness = { task_id: "t1", revision: 3, route: null, route_by: null, reasons: [], state: null, target: null,
  filled: [], risks: [], pending: false };

test("the route note says why, what was filled in, and when it is still looking", () => {
  expect(routeNote(base)).toBe("Not looked at yet.");
  expect(routeNote({ ...base, pending: true })).toBe("Looking at this task…");
  expect(routeNote({ ...base, state: "stale", route: "auto" })).toBe("Looking at this task…");
  expect(routeNote({ ...base, state: "unavailable" })).toContain("tried again");
  expect(routeNote({ ...base, route: "human", route_by: "owner", reasons: ["ignored"] })).toBe("You chose this.");
  expect(routeNote({ ...base, route: "plan_review", route_by: "readiness", state: "ready", reasons: ["Needs your plan review because it spends money."],
    filled: ["criteria", "target", "verify"], target: { id: "x", name: "acme", by: "readiness" } }))
    .toBe("Needs your plan review because it spends money. Filled in for you: draft success criteria, a repository, a verify command (acme).");
  expect(routeNote({ ...base, route: "direction", route_by: "readiness", state: "ready", reasons: ["Why."], filled: ["target"],
    target: { id: "x", name: "acme", by: "team" } })).toBe("Why. Filled in for you: a repository (acme, where this team's issues have run before).");
});
