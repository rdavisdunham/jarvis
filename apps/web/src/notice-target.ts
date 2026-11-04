import type { Notice } from "./types";

/** Where a notification opens. The push deep link (apps/api/jarvis/worker.py `notification_url`) and the
 * in-app Notifications page's Open button must agree: `notice-targets.json` holds the shared cases that
 * `notice-target.test.ts` and `tests/test_drift_guards.py` both check. Order matters: an agent
 * notification can carry a task too, but opens its run. */
export type NoticeDestination =
  | { open: "run"; run: string | null }
  | { open: "meetings"; source: string | null }
  | { open: "task"; task: string }
  | { open: "activity" }
  | { open: "today" }
  | { open: "notifications" };

export function noticeDestination(n: Pick<Notice, "task_id" | "target">): NoticeDestination {
  const target = n.target ?? {};
  if (target.view === "remotea") return { open: "run", run: target.run_id || null };
  if (target.view === "meetings")
    return { open: "meetings", source: (target as { source_id?: string }).source_id || null };
  if (n.task_id) return { open: "task", task: n.task_id };
  if (target.view === "activity") return { open: "activity" };
  if (target.view === "tasks") return { open: "today" };
  return { open: "notifications" };
}

/** What the Notifications page's Open does for each destination ("notifications": this list is already open; Today). */
export type NoticeOpeners = {
  run: (run: string | null) => void; meetings: (source: string | null) => void; task: (task: string) => void;
  activity: () => void; today: () => void;
};
export function openNotice(to: NoticeDestination, open: NoticeOpeners) {
  switch (to.open) {
    case "run": return open.run(to.run);
    case "meetings": return open.meetings(to.source);
    case "task": return open.task(to.task);
    case "activity": return open.activity();
    default: return open.today();
  }
}

const PRIORITY_LABELS: Record<string, string> = { urgent: "Urgent", high: "High priority", low: "Low priority" };

/** The line under a notification: its priority when not normal, and its state when worth saying.
 * Delivered (pushed to a device), read, snoozed and completed are separate states (notices.py `state`). */
export function noticeStatus(
  n: Pick<Notice, "state" | "importance" | "snoozed_until">,
  when: (iso: string) => string,
): { priority: string | null; state: string | null } {
  return {
    priority: PRIORITY_LABELS[n.importance ?? "normal"] ?? null,
    state:
      n.state === "snoozed" && n.snoozed_until
        ? "Snoozed until " + when(n.snoozed_until)
        : n.state === "delivered"
          ? "Delivered"
          : null,
  };
}
