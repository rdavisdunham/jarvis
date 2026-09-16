import { expect, it } from "vitest";
import { actionSchema } from "./site-actions";
import { validateSiteAction } from "./site-validation";
import { emptyOrganization } from "./productivity";
import { readView, viewStateSchema } from "./saved-views";
import { groupedTasks } from "./work-views";
import { matchesStatus } from "./workspace";
import { matchesTaskTab } from "./task-presets";
import type { Task } from "./types";

it("keeps Backlog in unfinished work, exact filters and dated views", () => {
  expect(matchesStatus("backlog", "active")).toBe(true);
  expect(matchesStatus("backlog", "backlog")).toBe(true);
  expect(matchesStatus("backlog", "open")).toBe(false);
  expect(matchesStatus("open", "backlog")).toBe(false);
  const task = { id: "later", title: "Later", status: "backlog",
    due_date: "2030-01-15", archived: false } as Task;
  const board = groupedTasks([task], "status");
  expect(board[0].key).toBe("backlog");
  expect(board[0].tasks).toEqual([task]);
  expect(matchesTaskTab(task, "today", "2030-01-15")).toBe(true);
});

it("restores Backlog saved links and accepts Eri's status filter", () => {
  const state = viewStateSchema.parse({ status: "backlog", layout: "board" });
  const params = new URLSearchParams({ view: "tasks", state: JSON.stringify(state) });
  expect(readView(params.toString())).toEqual(state);
  const action = actionSchema.parse({ id: "backlog", kind: "filter", status: "backlog" });
  expect(() => validateSiteAction(action, "all", emptyOrganization, "project")).not.toThrow();
});
