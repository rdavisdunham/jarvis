import { taskTabs, type TaskTab } from "./task-presets";
import { Tabs } from "./ux";
/** Text tabs for the Tasks page. Each tab's description is its tooltip. */
export function TaskTabs({value, onChange}: {value: TaskTab; onChange: (value: TaskTab) => void}) {
  return <Tabs id="task-tab" label="Task views" items={taskTabs} value={value} onChange={onChange} className="tabs task-tabs" panel="task-workspace" />;
}
