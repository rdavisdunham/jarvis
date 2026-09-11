export interface Task {
  id: string;
  title: string;
  notes: string;
  status: string;
  priority: number;
  project: string | null;
  due_date: string | null;
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
}
export interface Notice {
  id: string;
  title: string;
  body: string;
  scheduled_at: string;
  read_at: string | null;
  task_id: string | null;
  created_at: string;
}
export interface Memory {
  id: string;
  content: string;
  source_id: string;
  attribution: string;
  created_at: string;
  revision: number;
}
export interface Preferences {
  history_enabled: boolean;
  memory_learning: boolean;
  history_days: number;
  timezone: string;
  default_reminder_hour: number;
  monthly_budget_usd: number;
  detailed_notifications: boolean;
}
export interface Bootstrap {
  name: string;
  csrf: string;
  device_id: string;
  preferences: Preferences;
  budget: {
    spent_usd: number;
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
