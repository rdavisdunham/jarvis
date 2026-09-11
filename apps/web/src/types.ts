export interface Task {
  id: string;
  title: string;
  notes: string;
  status: string;
  priority: number;
  project: string | null;
  due_date: string | null;
  due_time: string | null;
  due_timezone: string | null;
  revision: number;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
  archived: boolean;
  occurrence_id: string | null;
}
export interface Schedule {
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
  result: { scanned?: number; merged?: number; queued_questions?: number } | null;
}
export interface Preferences {
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
  agent_model: string;
  name: string;
  csrf: string;
  device_id: string;
  preferences: Preferences;
  budget: {
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
  voice_options: Record<
    VoiceProvider,
    { label: string; voices: string[]; default_voice: string }
  >;
  last_backup_at: string | null;
  vapid_public_key: string;
  event_cursor: number;
}
export type View =
  | "today"
  | "inbox"
  | "week"
  | "all"
  | "reminders"
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
export interface UIAction {
  id: string;
  kind?: "show" | "chat" | "search" | "filter" | "form";
  view?: View;
  mode?: "open" | "close" | "auto";
  query?: string;
  status?: "all" | "open" | "completed";
  project?: string;
  form?: "task" | "reminder";
  entity_id?: string | null;
}

export interface UIContext {
  view: View;
  chat_open: boolean;
  mobile: boolean;
  voice_active: boolean;
  query: string;
  selected_task_id: string | null;
  visible_ids: string[];
  task_status: "all" | "open" | "completed";
  project: string;
}
