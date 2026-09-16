import { taskTabs, type TaskTab } from "./task-presets";
import { Tabs } from "./ux";
export function TaskTabs({value, onChange}: {value: TaskTab; onChange: (value: TaskTab) => void}) {
  return <div className="task-tab-header"><Tabs id="task-tab" label="Task views" items={taskTabs} value={value} onChange={onChange} className="task-tabs" panel="task-workspace" />
    <p className="task-tab-description">{taskTabs.find(t => t.id === value)?.description}</p></div>;
}
