export interface Task {
  space_id?: string | null;
  area_id?: string | null;
  planned_date?: string | null;
  estimate_minutes?: number | null;
  assignee_id?: string | null;
  id: string;
  title: string;
  notes: string;
  status: string;
  priority: number;
  project: string | null;
  project_id: string | null;
  parent_task_id: string | null;
  assignee: string;
  work_type: string;
  tags: string[];
  due_date: string | null;
  due_time: string | null;
  due_timezone: string | null;
  revision: number;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
  archived: boolean;
  occurrence_id: string | null;
  is_template?: boolean;
  external?: {
    provider?: string;
    identifier?: string;
    url?: string;
    team_id?: string;
    state?: string;
    state_id?: string;
    priority?: number;
    sync_state?: string;
    job_id?: string;
  };
}
export interface Project {
  space_id?: string | null;
  area_id?: string | null;
  status?: string;
  success_criteria?: string;
  start_date?: string | null;
  target_date?: string | null;
  goal_ids?: string[];
  task_count?: number;
  completed_task_count?: number;
  notes?: { id: string; title: string; archived?: boolean }[];
  id: string;
  name: string;
  description: string;
  archived: boolean;
  revision: number;
}
export interface CalendarEntry {
  timing?: "planned" | "deadline";
  space_id?: string | null;
  area_id?: string | null;
  id: string;
  entity_id: string;
  kind: "task" | "reminder" | "routine" | "google" | "event" | "block";
  description?: string;
  meeting_url?: string;
  end_at?: string;
  all_day?: boolean;
  recurring?: boolean;
  occurrence_start?: string | null;
  read_only?: boolean;
  calendar_title?: string;
  calendar_id?: string;
  url?: string | null;
  location?: string;
  busy?: boolean;
  conflicts?: string[];
  title: string;
  date: string;
  at: string | null;
  status: string;
  project_id: string | null;
  task_id: string | null;
  revision: number;
  projected: boolean;
  notification_id: string | null;
}
export interface Schedule {
  project_id: string | null;
  id: string;
  title: string;
  timezone: string;
  recurrence: string | null;
  kind: string;
  next_run_at: string | null;
  anchor_at: string;
  status: string;
  revision: number;
  task_id: string | null;
  completed_at: string | null;
}
export interface Notice {
  completed_at: string | null;
  schedule_id: string | null;
  id: string;
  title: string;
  body: string;
  scheduled_at: string;
  read_at: string | null;
  task_id: string | null;
  created_at: string;
}
export interface Memory {
  tags: string[];
  evidence: string;
  embedding_model: string | null;
  id: string;
  content: string;
  source_id: string;
  attribution: string;
  created_at: string;
  revision: number;
}
export interface MemoryReview {
  id: string;
  revision: number;
  question: string;
  candidates: Memory[];
}
export interface MemoryMaintenance {
  status: string;
  last_error: string | null;
  enabled: boolean;
  last_run_at: string | null;
  next_run_at: string;
  running: boolean;
  result: {
    scanned?: number;
    merged?: number;
    queued_questions?: number;
  } | null;
}
export type AgentProvider = "openai" | "gemini" | "groq";
export type AgentProfile = AgentProvider | "luna";
export interface Preferences {
  agent_profile: AgentProfile;
  agent_provider: AgentProvider;
  preferred_name: string;
  history_enabled: boolean;
  memory_learning: boolean;
  deep_sleep_enabled: boolean;
  history_days: number;
  timezone: string;
  default_reminder_hour: number;
  monthly_budget_usd: number;
  detailed_notifications: boolean;
}
export interface Bootstrap {
  agent_profile: AgentProfile;
  agent_reasoning: string | null;
  agent_provider: AgentProvider;
  agent_options: {
    id: AgentProfile;
    reasoning_effort: string | null;
    provider: AgentProvider;
    model: string;
    label: string;
    available: boolean;
  }[];
  agent_model: string;
  name: string;
  csrf: string;
  device_id: string;
  preferences: Preferences;
  budget:
    | { tracking_enabled: false; budget_mode: "disabled" }
    | {
        tracking_enabled?: true;
        spent_usd: number;
        uncertain_usd: number;
        active_reserved_usd: number;
        projected_month_usd: number;
        usage_by_model: Record<string, number>;
        budget_mode: string;
        reserved_usd: number;
        limit_usd: number;
        remaining_usd: number;
      };
  capabilities: {
    voice: boolean;
    chat: boolean;
    push: boolean;
    worker: boolean;
  };
  voice_options: Partial<
    Record<
      VoiceProvider,
      { label: string; voices: string[]; default_voice: string }
    >
  >;
  last_backup_at: string | null;
  vapid_public_key: string;
  event_cursor: number;
}
export type View =
  | "organize"
  | "today"
  | "inbox"
  | "week"
  | "all"
  | "calendar"
  | "reminders"
  | "notes"
  | "memory"
  | "notifications"
  | "settings";
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  pending?: boolean;
}

export type VoiceProvider = "realtime" | "live";
export type UIAction = import("zod").input<
  typeof import("./site-actions").actionSchema
>;

export interface UIContext {
  layout?: "list" | "board" | "timeline";
  sort?: string;
  group_by?: string;
  timeline_date?: string;
  timeline_span?: number;
  organization_tab?: string;
  settings_section?: string;
  notes_mode?: string;
  show_archived?: boolean;
  assignee?: string;
  work_type?: string;
  tag?: string;
  due_from?: string;
  due_through?: string;
  editor?: import("./editor-control").EditorSummary | null;
  device_preferences?: {
    voice: string;
    voices: string[];
    wake_enabled: boolean;
    wake_supported: boolean;
    density: string;
    private_chat: boolean;
  };

  space_id?: string;
  area_id?: string;
  goal_id?: string;
  view: View;
  calendar_date?: string;
  calendar_view?: "month" | "week" | "day";
  selected_schedule_id?: string | null;
  work_kind?: "all" | "task" | "reminder";
  chat_open: boolean;
  mobile: boolean;
  voice_active: boolean;
  query: string;
  selected_task_id: string | null;
  selected_task_ids?: string[];
  selected_calendar_event_id?: string | null;
  selected_note_id?: string | null;
  visible_ids: string[];
  task_status:
    | "active"
    | "all"
    | "open"
    | "in_progress"
    | "waiting"
    | "deferred"
    | "completed"
    | "cancelled";
  project: string;
}
