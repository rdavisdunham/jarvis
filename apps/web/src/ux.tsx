import { useEffect, useRef, type ReactNode } from "react";

export const humanLabel = (value: string) => value === "owner" ? "Me" : value.replaceAll("_", " ").replace(/^./, c => c.toUpperCase());
export const priorityLabels = ["None", "Low", "Medium", "High"];
export const plural = (count: number, noun: string) => `${count} ${noun}${count === 1 ? "" : "s"}`;
export function tabDestination(key: string, index: number, count: number) {
  return key === "ArrowRight" ? (index + 1) % count : key === "ArrowLeft" ? (index + count - 1) % count : key === "Home" ? 0 : key === "End" ? count - 1 : null;
}
export function Tabs<T extends string>({id, label, items, value, onChange, className = "", panel}: {
  id: string; label: string; items: readonly {id: T; label: ReactNode; description?: string}[]; value: T;
  onChange: (value: T) => unknown | Promise<unknown>; className?: string; panel: string;
}) {
  const root = useRef<HTMLDivElement>(null);
  return <div ref={root} className={className} role="tablist" aria-label={label}>
    {items.map((item, index) => <button type="button" key={item.id} id={`${id}-${item.id}`} role="tab" title={item.description}
      aria-controls={panel} aria-selected={value === item.id} tabIndex={value === item.id ? 0 : -1}
      onClick={() => void onChange(item.id)} onKeyDown={e => {
        const next = tabDestination(e.key, index, items.length);
        if (next === null) return;
        e.preventDefault();
        void Promise.resolve(onChange(items[next].id)).then(() => root.current?.querySelector<HTMLButtonElement>('[aria-selected="true"]')?.focus());
      }}>{item.label}</button>)}
  </div>;
}
let locks = 0, previousOverflow = "";
export function useBodyLock(active: boolean) {
  useEffect(() => {
    if (!active) return;
    if (locks++ === 0) { previousOverflow = document.body.style.overflow; document.body.style.overflow = "hidden"; }
    return () => { if (--locks === 0) document.body.style.overflow = previousOverflow; };
  }, [active]);
}
export function SchedulingHelp() {
  return <details className="context-help"><summary>Dates, reminders & time blocks</summary><dl>
    <dt>Planned day</dt><dd>When you intend to work on a task.</dd>
    <dt>Deadline</dt><dd>When it must be finished. A deadline does not send an alert.</dd>
    <dt>Reminder</dt><dd>A notification for the same task, once or on a repeating schedule.</dd>
    <dt>Time block</dt><dd>Time reserved on your calendar to do the work.</dd>
  </dl></details>;
}
export function PlannerGuide() {
  return <details className="planner-guide"><summary>A quick guide to your planner</summary>
    <p>Example only: these records are never added to your workspace.</p>
    <ol><li><strong>Goal:</strong> Run a half marathon — an outcome, measured by finishing the race.</li>
      <li><strong>Project:</strong> Complete a 12-week training plan — a defined piece of work that supports the goal.</li>
      <li><strong>Task:</strong> Run 5 km on Tuesday — one action. Plan it for Tuesday and reserve a morning time block.</li>
      <li><strong>Note:</strong> Training log — observations linked to the project and its tasks.</li></ol>
    <p>Projects can support several goals; goals can draw on several projects. One-off tasks need neither. Spaces such as Work and Personal classify records; workspaces determine who can see them.</p>
    <SchedulingHelp />
  </details>;
}
