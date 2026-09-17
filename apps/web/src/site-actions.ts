import { z } from "zod";

export const views = [
  "organize",
  "today",
  "inbox",
  "week",
  "all",
  "reminders",
  "calendar",
  "notes",
  "memory",
  "notifications",
  "settings",
] as const;
export const editorKinds = [
  "task",
  "reminder",
  "note",
  "goal",
  "project",
  "area",
  "space",
  "actor",
  "event",
  "google_event",
  "bulk",
  "memory",
] as const;
export const taskStates = [
  "all",
  "active",
  "backlog",
  "open",
  "in_progress",
  "waiting",
  "deferred",
  "completed",
  "cancelled",
] as const;
const date = z.string().date();
const scalarPatchValue = z.union([
  z.string().max(30000),
  z.number().finite(),
  z.boolean(),
  z.null(),
  z.array(z.string().max(1000)).max(200),
]);
const patchValue = z.union([scalarPatchValue, z.record(z.string().max(80), scalarPatchValue).refine(v => Object.keys(v).length <= 60, "Too many fields")]);
export const actionSchema = z
  .object({
    id: z.string().max(150),
    kind: z
      .enum([
        "records",
        "show",
        "chat",
        "activity",
        "search",
        "filter",
        "form",
        "calendar",
        "select",
        "workspace",
        "editor",
        "device",
        "saved_view",
      ])
      .default("show"),
    view: z.enum(views).optional(),
    view_operation: z.enum(["list", "save", "load", "delete"]).optional(),
    view_name: z.string().max(80).optional(),
    saved_view_id: z.string().max(36).optional(),
    type_id: z.string().max(80).optional(),
    parent_id: z.string().max(36).optional(),
    record_id: z.string().max(36).optional(),
    proposal_id: z.string().max(36).optional(),
    record_group: z.string().max(80).optional(),
    field: z.string().max(80).optional(),
    value: z.string().max(300).optional(),
    entity_id: z.string().nullable().optional(),
    mode: z.enum(["open", "close", "auto"]).optional(),
    query: z.string().max(300).optional(),
    status: z.enum(taskStates).optional(),
    project: z.string().max(200).optional(),
    project_id: z.string().max(36).optional(),
    space_id: z.string().max(36).optional(),
    area_id: z.string().max(36).optional(),
    goal_id: z.string().max(36).optional(),
    assignee: z.string().max(100).optional(),
    work_type: z.string().max(80).optional(),
    tag: z.string().max(40).optional(),
    due_from: z.union([date, z.literal("")]).optional(),
    due_through: z.union([date, z.literal("")]).optional(),
    date: date.optional(),
    open_details: z.boolean().optional(),
    calendar_view: z.enum(["month", "week", "day"]).optional(),
    work_kind: z.enum(["all", "task", "reminder"]).optional(),
    task_ids: z.array(z.string()).max(100).optional(),
    form: z.enum(editorKinds).optional(),
    layout: z.enum(["list", "board", "timeline"]).optional(),
    sort: z.enum(["priority", "due", "planned", "title", "updated"]).optional(),
    group_by: z.enum(["status", "project", "assignee"]).optional(),
    timeline_date: date.optional(),
    timeline_span: z
      .union([z.literal(14), z.literal(30), z.literal(90)])
      .optional(),
    organization_tab: z
      .enum(["goal", "project", "area", "space", "actor"])
      .optional(),
    settings_section: z
      .enum([
        "profile",
        "voice",
        "integrations",
        "privacy",
        "system",
        "sharing",
      ])
      .optional(),
    notes_mode: z.enum(["keyword", "semantic"]).optional(),
    show_archived: z.boolean().optional(),
    operation: z.enum(["read", "patch", "save", "close", "discard"]).optional(),
    changes: z.record(z.string(), patchValue).optional(),
    voice: z.string().max(50).optional(),
    wake_enabled: z.boolean().optional(),
    density: z.enum(["compact", "comfortable"]).optional(),
  })
  .strict();
