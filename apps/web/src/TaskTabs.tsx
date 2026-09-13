import { taskTabs, type TaskTab } from "./task-presets";
export function TaskTabs({
  value,
  onChange,
}: {
  value: TaskTab;
  onChange: (value: TaskTab) => void;
}) {
  return (
    <div className="task-tab-header">
      <div className="task-tabs" role="tablist" aria-label="Task views">
        {taskTabs.map((tab, index) => (
          <button
            key={tab.id}
            id={"task-tab-" + tab.id}
            role="tab"
            type="button"
            aria-selected={value === tab.id}
            aria-controls="task-workspace"
            tabIndex={value === tab.id ? 0 : -1}
            onClick={() => onChange(tab.id)}
            onKeyDown={(event) => {
              const next =
                event.key === "ArrowRight"
                  ? (index + 1) % taskTabs.length
                  : event.key === "ArrowLeft"
                    ? (index + taskTabs.length - 1) % taskTabs.length
                    : event.key === "Home"
                      ? 0
                      : event.key === "End"
                        ? taskTabs.length - 1
                        : null;
              if (next !== null) {
                event.preventDefault();
                onChange(taskTabs[next].id);
                document
                  .getElementById("task-tab-" + taskTabs[next].id)
                  ?.focus();
              }
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <p className="task-tab-description">
        {taskTabs.find((tab) => tab.id === value)?.description}
      </p>
    </div>
  );
}
