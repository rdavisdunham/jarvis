import { createContext, useContext } from "react";
import type { Schedule, Task } from "./types";
import type { Actor } from "./productivity";
import type { AccountContext } from "./assignees";

/** What the app gives every task card (TaskDetails, RecordCard, TaskDialog) without prop drilling. */
export type TaskCardHost = {
  /** The signed-in account, for "Davis (me)" and shared-workspace members. */
  account?: AccountContext;
  actors?: Actor[];
  tasks: Task[];
  schedules: Schedule[];
  openNote: (id: string) => void;
  newNote: (task: Task) => void;
  reminder: (schedule: Schedule | null, task: Task) => void;
  reserveTime: (task: Task) => void;
};
export const TaskCardHostContext = createContext<TaskCardHost | null>(null);
export const useTaskCardHost = () => useContext(TaskCardHostContext);
