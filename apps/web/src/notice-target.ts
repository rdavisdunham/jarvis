import type { Notice } from "./types";

/** Where a notification opens. The push deep link (apps/api/jarvis/worker.py `notification_url`) and the
 * in-app Notifications page's Open button must agree: `notice-targets.json` holds the shared cases that
 * `notice-target.test.ts` and `tests/test_drift_guards.py` both check. Order matters: an agent
 * notification can carry a task too, but opens its run. */
export type NoticeDestination =
  | { open: "run"; run: string | null }
  | { open: "task"; task: string }
  | { open: "activity" }
  | { open: "today" }
  | { open: "notifications" };

export function noticeDestination(n: Pick<Notice, "task_id" | "target">): NoticeDestination {
  const target = n.target ?? {};
  if (target.view === "remotea") return { open: "run", run: target.run_id || null };
  if (n.task_id) return { open: "task", task: n.task_id };
  if (target.view === "activity") return { open: "activity" };
  if (target.view === "tasks") return { open: "today" };
  return { open: "notifications" };
}
