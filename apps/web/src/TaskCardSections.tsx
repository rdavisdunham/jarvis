/**
 * Sections shared by the two task cards (RecordCard.tsx and TaskDetails.tsx). Each card supplies
 * its own data and save path; the markup and labels live here once so the cards cannot drift.
 * The labels are TASK_CARD_SECTIONS (task-card-sections.ts); task-card-parity.test.tsx checks both
 * cards render every one of them. See docs/TASK_CARDS.md.
 */
import { createContext, lazy, Suspense, useContext, useEffect, useState } from "react";
import { Bell, ChevronRight, CornerLeftUp, FileText, Plus } from "lucide-react";
import { api } from "./api";
import { timeLabel, recurrenceLabel } from "./components";
import { humanLabel, SchedulingHelp } from "./ux";
import { SaveAsTemplateDialog } from "./Templates";
import { TASK_CARD_SECTIONS } from "./task-card-sections";
import type { Schedule, Task } from "./types";
import type { NoteRecord } from "./Notes";
import type { CustomRecord, Schema, SchemaType } from "./structure-types";

const RunTask = lazy(() => import("./Remotea").then((m) => ({ default: m.RunTask })));

/**
 * What the shell (App.tsx) lends a task card that it cannot reach on its own: the loaded task and
 * reminder lists and the editors that live outside the card. TaskDetails gets the same things as
 * props; RecordCard, which Structure.tsx renders, reads them here. Each opener is called after the
 * card has closed itself.
 */
export type TaskCardHost = {
  tasks: Task[];
  schedules: Schedule[];
  openNote: (id: string) => void;
  newNote: (task: Task) => void;
  reminder: (schedule: Schedule | null, task: Task) => void;
  reserveTime: (task: Task) => void;
};
export const TaskCardHostContext = createContext<TaskCardHost | null>(null);
export const useTaskCardHost = () => useContext(TaskCardHostContext);

/** The remotea Run control (Remotea.tsx `RunTask`, loaded on demand). It hides itself when remotea is off. */
export function TaskRun({ taskId, canEdit, onOpenRun }: { taskId: string; canEdit: boolean; onOpenRun?: (run: string) => void }) {
  return <Suspense fallback={null}><RunTask taskId={taskId} canEdit={canEdit} onOpenRun={onOpenRun}/></Suspense>;
}

type BugReportSummary = { id: string; status: string; created_at: string; screenshot: { bytes: number } | null; screenshot_error: string | null };
/** The report behind a task filed by Report a bug (BugReport.tsx): its screenshot inline, and its status.
 * Renders nothing unless the task is tagged `bug-report`. */
export function BugReportDetails({ taskId, tags }: { taskId: string; tags: readonly string[] }) {
  const tagged = tags.includes("bug-report");
  const [report, setReport] = useState<BugReportSummary | null | undefined>(undefined);
  useEffect(() => {
    if (!tagged) return;
    let live = true;
    api<{ items: BugReportSummary[] }>("/bug-reports?task_id=" + encodeURIComponent(taskId))
      .then((r) => { if (live) setReport(r.items[0] ?? null); }, () => { if (live) setReport(null); });
    return () => { live = false; };
  }, [taskId, tagged]);
  if (!tagged) return null;
  const shot = report ? "/api/v1/bug-reports/" + report.id + "/screenshot" : "";
  return <section className="detail-section" aria-label={TASK_CARD_SECTIONS.bugReport}>
    <div className="detail-section-head"><h3>Bug report</h3>{report && <span className="chip">{humanLabel(report.status)}</span>}</div>
    {report === undefined ? <p className="footnote">Loading the report…</p>
      : report === null ? <p className="footnote">The report is not available here (deleted, or filed from another workspace).</p>
      : report.screenshot ? <a className="bug-report-task-shot" href={shot} target="_blank" rel="noreferrer">
        <img src={shot} alt="Screenshot attached to this bug report" loading="lazy"/></a>
      : <p className="footnote">No screenshot{report.screenshot_error ? ": " + report.screenshot_error : ""}.</p>}
  </section>;
}

/** Parent task and subtasks. Renders nothing when there are neither. */
export function RelatedTasks({ parent, subtasks, onOpen }: { parent?: Task; subtasks: Task[]; onOpen: (task: Task) => void }) {
  if (!parent && !subtasks.length) return null;
  return <section className="detail-section" aria-label={TASK_CARD_SECTIONS.relatedTasks}>
    <div className="detail-section-head"><h3>Related tasks</h3></div>
    {parent && <button className="detail-link-row" onClick={() => onOpen(parent)}>
      <CornerLeftUp size={15}/><span className="detail-link-title">{parent.title}</span><span className="chip">Parent</span>
    </button>}
    {subtasks.map((t) => <button key={t.id} className="detail-link-row" onClick={() => onOpen(t)}>
      <ChevronRight size={15}/><span className="detail-link-title">{t.title}</span>
      <span className={"chip" + (t.status === "completed" ? " chip-done" : "")}>{humanLabel(t.status)}</span>
    </button>)}
  </section>;
}

/** The new-subtask form. `onAdd` saves; the field clears once it resolves and keeps the text if it throws. */
export function AddSubtask({ noun = "task", disabled, onAdd }: { noun?: string; disabled: boolean; onAdd: (title: string) => Promise<unknown> }) {
  const [title, setTitle] = useState(""), [saving, setSaving] = useState(false);
  return <form className="org-add detail-subtask" aria-label={TASK_CARD_SECTIONS.addSubtask} onSubmit={(e) => {
    e.preventDefault();
    if (!title.trim() || saving) return;
    setSaving(true);
    void onAdd(title.trim()).then(() => setTitle("")).catch(() => {}).finally(() => setSaving(false));
  }}>
    <input aria-label="New subtask" placeholder={"Add " + noun.toLowerCase() + " inside…"} value={title} maxLength={500} disabled={disabled || saving} onChange={(e) => setTitle(e.target.value)}/>
    <button className="btn btn-soft btn-sm" disabled={disabled || saving || !title.trim()}>Add subtask</button>
  </form>;
}

/** Notes linked to a task, with New linked note. */
export function LinkedNotes({ taskId, disabled, onOpen, onNew }: { taskId: string; disabled: boolean; onOpen: (id: string) => void; onNew: () => void }) {
  const [notes, setNotes] = useState<NoteRecord[]>([]), [error, setError] = useState("");
  useEffect(() => {
    let live = true;
    setError("");
    void api<{ items: NoteRecord[] }>("/notes?task_id=" + encodeURIComponent(taskId))
      .then((r) => { if (live) setNotes(r.items); })
      .catch(() => { if (live) setError("Linked notes could not be loaded."); });
    return () => { live = false; };
  }, [taskId]);
  return <section className="detail-section" aria-label={TASK_CARD_SECTIONS.linkedNotes}>
    <div className="detail-section-head">
      <h3>Linked notes</h3>
      {!!notes.length && <span className="detail-count">{notes.length}</span>}
      <button className="btn btn-ghost btn-sm" disabled={disabled} onClick={onNew}><Plus size={15}/>New linked note</button>
    </div>
    {error && <p role="alert" className="field-error">{error}</p>}
    {notes.length
      ? notes.map((n) => <button className="detail-link-row" key={n.id} onClick={() => onOpen(n.id)}>
          <FileText size={15}/><span className="detail-link-title">{n.title}</span>
        </button>)
      : <p className="detail-empty">No linked notes yet.</p>}
  </section>;
}

/** Reminders for the task, Add reminder, and Reserve time (a calendar work block). */
export function TaskSchedule({ reminders, isTemplate, disabled, onReminder, onReserve }: {
  reminders: Schedule[]; isTemplate: boolean; disabled: boolean; onReminder: (schedule: Schedule | null) => void; onReserve: () => void;
}) {
  return <section className="detail-section" aria-label={TASK_CARD_SECTIONS.schedule}>
    <div className="detail-section-head">
      <h3>Reminders and time</h3>
      {!!reminders.length && <span className="detail-count">{reminders.length}</span>}
      <button className="btn btn-ghost btn-sm" disabled={disabled} onClick={() => onReminder(null)}><Plus size={15}/>Add reminder</button>
    </div>
    {reminders.length
      ? reminders.map((s) => <div className="detail-link-row" key={s.id}>
          <button className="detail-link-open" onClick={() => onReminder(s)}><Bell size={15}/><span className="detail-link-title">{s.title}</span></button>
          <span className="detail-meta">
            {s.next_run_at && <span className="chip">{timeLabel(s.next_run_at, s.timezone)}</span>}
            <span className="chip">{recurrenceLabel(s.recurrence)}</span>
            {s.status !== "active" && <span className="chip">{humanLabel(s.status)}</span>}
          </span>
        </div>)
      : <p className="detail-empty">No reminders for this task.</p>}
    {!isTemplate && <div className="detail-actions">
      <button className="btn btn-soft btn-sm" disabled={disabled} onClick={onReserve}>Reserve time for this task</button>
    </div>}
    <SchedulingHelp/>
  </section>;
}

/**
 * Save as template, with its dialog. `row` is the task's record (null while it loads: the button
 * shows, disabled). Without `type` the record's type is read from the schema when clicked.
 * `beforeOpen` lets the card finish an edit in progress first.
 */
export function SaveTemplate({ row, type, hasContents, disabled, beforeOpen, onNotice }: {
  row: CustomRecord | null; type?: SchemaType; hasContents: boolean; disabled: boolean;
  beforeOpen?: () => Promise<unknown>; onNotice: (message: string) => void;
}) {
  const [open, setOpen] = useState<SchemaType | null>(null), [loading, setLoading] = useState(false);
  const start = async () => {
    if (!row) return;
    setLoading(true);
    // A failed save in progress is already shown by the card; stay put.
    try { await beforeOpen?.(); } catch { setLoading(false); return; }
    try {
      const found = type ?? (await api<Schema>("/structure")).types.find((t) => t.id === row.type_id);
      if (!found) throw new Error("This record's type is unavailable.");
      setOpen(found);
    } catch (e) {
      onNotice((e as Error).message);
    } finally {
      setLoading(false);
    }
  };
  return <>
    <button className="btn btn-ghost btn-sm" aria-label={TASK_CARD_SECTIONS.saveTemplate} disabled={disabled || loading || !row} onClick={() => void start()}>Save as template</button>
    {open && row && <SaveAsTemplateDialog row={row} type={open} hasContents={hasContents} onClose={() => setOpen(null)}
      onSaved={(tpl) => { setOpen(null); onNotice("Saved template " + tpl.name + ". Find it under Start from template."); }}/>}
  </>;
}
