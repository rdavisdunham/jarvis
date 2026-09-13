import type { Project, Task } from "./types";

export interface Home { space_id?: string | null; area_id?: string | null }
export interface Space { id: string; name: string; description: string; archived: boolean; revision: number }
export interface Area extends Space { space_id: string }
export interface NoteLink { id: string; title: string; archived?: boolean }
export interface Goal extends Space, Home {
  success_criteria: string;
  status: "planned" | "active" | "on_hold" | "achieved" | "abandoned";
  horizon: "unspecified" | "short_term" | "long_term";
  parent_goal_id: string | null;
  target_date: string | null;
  metric_unit: string;
  metric_baseline: number;
  metric_current: number | null;
  metric_target: number | null;
  progress: number | null;
  project_ids: string[];
  notes: NoteLink[];
}
export interface Actor { id: string; name: string; kind: "person" | "agent"; archived: boolean; revision: number }
export interface Organization { spaces: Space[]; areas: Area[]; goals: Goal[]; projects: Project[]; actors: Actor[] }
export const emptyOrganization: Organization = { spaces: [], areas: [], goals: [], projects: [], actors: [] };
export interface OrganizationFilter { space: string; area: string; goal: string }
export const emptyFilter: OrganizationFilter = { space: "", area: "", goal: "" };
export function matchesOrganization(item: Home & { project_id?: string | null }, filter: OrganizationFilter, org: Organization) {
  const project = org.projects.find(p => p.id === item.project_id);
  const space = project ? project.space_id : item.space_id;
  const area = project ? project.area_id : item.area_id;
  return (!filter.space || space === filter.space) && (!filter.area || area === filter.area) &&
    (!filter.goal || !!project?.goal_ids?.includes(filter.goal));
}
export function scheduledBy(task: Pick<Task, "due_date" | "planned_date">, day: string) {
  return !!((task.planned_date && task.planned_date <= day) || (task.due_date && task.due_date <= day));
}
