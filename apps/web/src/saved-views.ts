import { z } from "zod";
export const viewStateSchema = z
  .object({
    tab: z.enum(["today", "inbox", "week", "all"]).default("all"),
    query: z.string().max(300).default(""),
    status: z
      .enum([
        "all",
        "active",
        "open",
        "in_progress",
        "waiting",
        "deferred",
        "completed",
        "cancelled",
      ])
      .default("active"),
    project: z.string().max(200).default(""),
    space: z.string().max(36).default(""),
    area: z.string().max(36).default(""),
    goal: z.string().max(36).default(""),
    assignee: z.string().max(100).default(""),
    work_type: z.string().max(80).default(""),
    tag: z.string().max(40).default(""),
    due_from: z.string().date().or(z.literal("")).default(""),
    due_through: z.string().date().or(z.literal("")).default(""),
    kind: z.enum(["all", "task", "reminder"]).default("all"),
    layout: z.enum(["list", "board", "timeline"]).default("list"),
    sort: z
      .enum(["priority", "due", "planned", "title", "updated"])
      .default("priority"),
    group: z.enum(["status", "project", "assignee"]).default("status"),
    timeline_date: z.string().date().or(z.literal("")).default(""),
    timeline_span: z
      .union([z.literal(14), z.literal(30), z.literal(90)])
      .default(30),
  })
  .strict();
export type ViewState = z.infer<typeof viewStateSchema>;
export type SavedView = {
  id: string;
  name: string;
  revision: number;
  state: ViewState;
};
export function readView(search: string): ViewState | null {
  try {
    const params = new URLSearchParams(search);
    if (params.get("view") !== "tasks") return null;
    const raw = params.get("state");
    if (!raw || raw.length > 5000) return null;
    const parsed = viewStateSchema.safeParse(JSON.parse(raw));
    return parsed.success ? parsed.data : null;
  } catch {
    return null;
  }
}
export function viewLink(state: ViewState) {
  const url = new URL(location.href);
  url.searchParams.set("view", "tasks");
  url.searchParams.set("tab", state.tab);
  url.searchParams.set("state", JSON.stringify(state));
  return url;
}
