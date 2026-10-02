import { useEffect, useRef, useState, type FormHTMLAttributes, type HTMLAttributes, type ReactNode, type RefObject } from "react";

export const humanLabel = (value: string) => value === "owner" ? "Me" : value.replaceAll("_", " ").replace(/^./, c => c.toUpperCase());
export const priorityLabels = ["None", "Low", "Medium", "High"];
export const plural = (count: number, noun: string) => `${count} ${noun}${count === 1 ? "" : "s"}`;
export function tabDestination(key: string, index: number, count: number) {
  return key === "ArrowRight" ? (index + 1) % count : key === "ArrowLeft" ? (index + count - 1) % count : key === "Home" ? 0 : key === "End" ? count - 1 : null;
}
export function Tabs<T extends string>({id, label, items, value, onChange, className = "", panel, orientation = "horizontal"}: {
  id: string; label: string; items: readonly {id: T; label: ReactNode; description?: string}[]; value: T;
  onChange: (value: T) => unknown | Promise<unknown>; className?: string; panel: string; orientation?: "horizontal" | "vertical";
}) {
  const root = useRef<HTMLDivElement>(null);
  return <div ref={root} className={className} role="tablist" aria-label={label} aria-orientation={orientation}>
    {items.map((item, index) => <button type="button" key={item.id} id={`${id}-${item.id}`} role="tab" title={item.description}
      aria-controls={panel} aria-selected={value === item.id} tabIndex={value === item.id ? 0 : -1}
      onClick={() => void onChange(item.id)} onKeyDown={e => {
        const key = orientation === "vertical" ? ({ArrowDown: "ArrowRight", ArrowUp: "ArrowLeft"} as Record<string,string>)[e.key] ?? e.key : e.key;
        const next = tabDestination(key, index, items.length);
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
// ---- Breakpoints -----------------------------------------------------------------------
// The single source of truth for layout tiers used from JS. CSS cannot import these, so the
// matching `@media (max-width: …px)` rules in style.css / ux.css / shell.css use the same
// numbers: COMPACT (≤1000: conversation overlays, search collapses), PHONE (≤700: navigation
// becomes a drawer) and NARROW (≤600: small-screen tweaks).
export const COMPACT_MAX_WIDTH = 1000;
export const PHONE_MAX_WIDTH = 700;
export const NARROW_MAX_WIDTH = 600;
export const maxWidthQuery = (px: number) => `(max-width: ${px}px)`;
export function matchesMaxWidth(px: number) {
  return typeof matchMedia === "function" ? matchMedia(maxWidthQuery(px)).matches : false;
}
export function useMaxWidth(px: number) {
  const [matches, setMatches] = useState(() => matchesMaxWidth(px));
  useEffect(() => {
    if (typeof matchMedia !== "function") return;
    const query = matchMedia(maxWidthQuery(px));
    const update = () => setMatches(query.matches);
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, [px]);
  return matches;
}
// ---- Dialogs -------------------------------------------------------------------------
// Dialogs stay in the normal stacking context (not native showModal()) on purpose: the
// Eri conversation panel sits above open dialogs and must stay usable while the agent
// drives a form. Focus is still managed: initial focus moves into the dialog, Tab wraps
// inside the topmost dialog, Escape is offered to the topmost dialog only, and focus
// returns to the opener on close.
const FOCUSABLE = 'button:not(:disabled), input:not(:disabled), textarea:not(:disabled), select:not(:disabled), a[href], summary, [tabindex="0"]';
const dialogStack: HTMLElement[] = [];
export const topDialog = () => dialogStack.at(-1) ?? null;
/** Where Tab should move focus, or null to let the browser handle it. */
export function trapTarget<T>(items: T[], active: T | null, shift: boolean, inside: boolean, idle: boolean): T | null {
  if (!items.length) return null;
  const first = items[0], last = items[items.length - 1];
  if (!inside) return idle ? (shift ? last : first) : null;
  if (shift && active === first) return last;
  if (!shift && active === last) return first;
  return null;
}
function newestDialog() {
  return Array.from(document.querySelectorAll<HTMLElement>('[role="dialog"]'))
    .filter(el => el.getClientRects().length && !dialogStack.includes(el)).at(-1) ?? null;
}
export function useDialogFocus(options: { ref?: RefObject<HTMLElement | null>; onEscape?: () => void } = {}) {
  useBodyLock(true);
  const escape = useRef(options.onEscape);
  escape.current = options.onEscape;
  const ref = options.ref;
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    let dialog = ref?.current ?? newestDialog();
    if (dialog) {
      dialogStack.push(dialog);
      // Don't pull focus out of the conversation when the agent opens a form mid-chat.
      if (!dialog.contains(document.activeElement) && !previous?.closest(".companion")) {
        const preferred = dialog.querySelector<HTMLElement>("[autofocus], [data-autofocus]");
        if (preferred) preferred.focus();
        else {
          if (!dialog.hasAttribute("tabindex")) dialog.tabIndex = -1;
          dialog.focus({ preventScroll: true });
        }
      }
    }
    const keydown = (event: KeyboardEvent) => {
      // Legacy callers without a ref may render their dialog element after mounting.
      if (!dialog && (dialog = newestDialog())) dialogStack.push(dialog);
      if (event.defaultPrevented || !dialog || topDialog() !== dialog) return;
      if (event.key === "Escape" && escape.current) {
        event.preventDefault();
        escape.current();
        return;
      }
      if (event.key !== "Tab") return;
      const items = Array.from(dialog.querySelectorAll<HTMLElement>(FOCUSABLE))
        .filter(el => el.getClientRects().length && el.tabIndex >= 0);
      const active = document.activeElement as HTMLElement | null;
      const target = trapTarget(items, active, event.shiftKey, dialog.contains(active),
        !active || active === document.body || !!active.closest(".modal-backdrop"));
      if (target) {
        event.preventDefault();
        target.focus();
      }
    };
    document.addEventListener("keydown", keydown);
    return () => {
      document.removeEventListener("keydown", keydown);
      const index = dialog ? dialogStack.lastIndexOf(dialog) : -1;
      if (index >= 0) dialogStack.splice(index, 1);
      if (previous?.isConnected) previous.focus({ preventScroll: true });
    };
  }, []);
}
type DialogOwnProps = {
  /** Called when the backdrop itself (not the dialog) is clicked. */
  onBackdrop?: () => void;
  /** Only for dialogs whose Escape is not already handled by the app shell. */
  onEscape?: () => void;
  backdropClassName?: string;
  /** Optional ref to the dialog element itself. */
  dialogRef?: RefObject<HTMLElement | null>;
  children?: ReactNode;
};
type DialogProps = DialogOwnProps & (
  | ({ as?: "section" } & HTMLAttributes<HTMLElement>)
  | ({ as: "form" } & FormHTMLAttributes<HTMLFormElement>)
);
/** Shared modal shell: backdrop + `role="dialog"` element with managed focus. */
export function Dialog({ as = "section", onBackdrop, onEscape, backdropClassName, dialogRef, children, ...rest }: DialogProps) {
  const own = useRef<HTMLElement>(null);
  const ref = dialogRef ?? own;
  useDialogFocus({ ref, onEscape });
  const Tag = as as "section";
  return <div className={"modal-backdrop" + (backdropClassName ? " " + backdropClassName : "")}
    onClick={onBackdrop ? event => { if (event.target === event.currentTarget) onBackdrop(); } : undefined}>
    <Tag ref={ref as RefObject<HTMLElement>} role="dialog" aria-modal="true" {...(rest as HTMLAttributes<HTMLElement>)}>{children}</Tag>
  </div>;
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
