import type { UIAction, View } from "./types";
import type { Organization } from "./productivity";

const fields: Record<string, string[]> = {
  show: ["view", "entity_id"],
  chat: ["mode"],
  search: ["query", "view"],
  filter: [
    "view",
    "status",
    "project",
    "project_id",
    "space_id",
    "area_id",
    "goal_id",
    "work_kind",
    "assignee",
    "work_type",
    "tag",
    "due_from",
    "due_through",
  ],
  form: ["form", "entity_id"],
  calendar: ["date", "calendar_view", "entity_id"],
  select: ["task_ids"],
  workspace: [
    "view",
    "layout",
    "sort",
    "group_by",
    "timeline_date",
    "timeline_span",
    "organization_tab",
    "settings_section",
    "notes_mode",
    "show_archived",
  ],
  editor: ["operation", "changes"],
  device: ["voice", "wake_enabled", "density", "private_chat"],
};
export function validateSiteAction(
  action: UIAction,
  view: View,
  organization: Organization,
  organizationTab: string,
) {
  const kind = action.kind ?? "show";
  for (const key of Object.keys(action))
    if (!["id", "kind", ...fields[kind]].includes(key))
      throw new Error(key + " is not supported by " + kind + ".");
  const work = ["all", "today", "week", "inbox", "calendar", "reminders"];
  const target =
    action.view ??
    (kind === "filter" && ![...work, "notes", "organize"].includes(view)
      ? "all"
      : view);
  if (kind === "filter") {
    if (![...work, "notes", "organize"].includes(target))
      throw new Error(
        "This page has no record filters. Use search or choose a workspace.",
      );
    const allowed =
      target === "organize"
        ? ["space_id"]
        : target === "notes"
          ? ["project", "project_id", "space_id", "area_id", "goal_id"]
          : fields.filter;
    for (const key of Object.keys(action))
      if (!["id", "kind", "view", ...allowed].includes(key))
        throw new Error(key + " does not filter " + target + ".");
    for (const [field, rows] of [
      ["project_id", organization.projects],
      ["space_id", organization.spaces],
      ["area_id", organization.areas],
      ["goal_id", organization.goals],
    ] as const) {
      const id = action[field];
      if (id && !rows.some((r) => r.id === id))
        throw new Error(
          "That " +
            field.replace("_id", "") +
            " is unavailable. Read current organization records.",
        );
    }
    if (
      action.project &&
      !organization.projects.some((p) => p.name === action.project)
    )
      throw new Error("That project is unavailable.");
    if (
      action.project &&
      action.project_id &&
      !organization.projects.some(
        (p) => p.id === action.project_id && p.name === action.project,
      )
    )
      throw new Error("Project name and ID refer to different records.");
    if (
      action.due_from &&
      action.due_through &&
      action.due_from > action.due_through
    )
      throw new Error("Due-from must be on or before due-through.");
  }
  if (kind === "workspace") {
    const isWork = ["all", "today", "week", "inbox"].includes(target),
      isProjects =
        target === "organize" &&
        (action.organization_tab ?? organizationTab) === "project";
    if (action.layout && !isWork && !isProjects)
      throw new Error(
        "List, board and timeline are available in Work and the Projects tab.",
      );
    if ((action.sort || action.group_by) && !isWork)
      throw new Error("Task sorting/grouping belongs to a Work view.");
    if (
      (action.timeline_date || action.timeline_span) &&
      !isWork &&
      !isProjects
    )
      throw new Error("Choose a Work or Projects timeline.");
    if (action.organization_tab && target !== "organize")
      throw new Error("Organization tabs belong to Projects & goals.");
    if (action.settings_section && target !== "settings")
      throw new Error("Choose Settings for that section.");
    if (action.notes_mode && target !== "notes")
      throw new Error("Choose Notes for search mode.");
    if (
      action.show_archived !== undefined &&
      !["notes", "organize"].includes(target)
    )
      throw new Error(
        "Archived collections are available in Notes and Projects & goals.",
      );
  }
  if (kind === "editor" && action.changes && action.operation !== "patch")
    throw new Error("Use patch to fill fields.");
  if (
    kind === "form" &&
    ["goal", "project", "area", "space", "actor"].includes(action.form ?? "") &&
    action.entity_id
  ) {
    const collections = {
      goal: organization.goals,
      project: organization.projects,
      area: organization.areas,
      space: organization.spaces,
      actor: organization.actors,
    };
    if (
      !collections[action.form as keyof typeof collections].some(
        (r) => r.id === action.entity_id,
      )
    )
      throw new Error(
        "That record is unavailable. Refresh organization records.",
      );
  }
}
