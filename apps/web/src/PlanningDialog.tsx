import { z } from "zod";
import { useEditor, nullableId, choice } from "./editor-control";
import { useEffect, useState } from "react";
import { X } from "lucide-react";
import { api } from "./api";
import { useDialogFocus } from "./components";
import { localDateTime, shiftDate } from "./workspace";
import type { CalendarEntry, Task } from "./types";
import type { GoogleStatus } from "./GoogleSettings";
type Fields = {
  title: string;
  start: string;
  end: string;
  timezone: string;
  all_day: boolean;
  location: string;
  description: string;
  busy: boolean;
};
type Record = {
  id: string;
  revision: number;
  kind: "event" | "block";
  task_id: string | null;
  fields: Fields;
  google_calendar_id: string | null;
  google_state: string;
  google_job_id: string | null;
  write_message?: string;
  write_status?: string;
};
export function PlanningDialog({
  event,
  tasks,
  timezone,
  mutate,
  onClose,
  onSaved,
}: {
  event: CalendarEntry;
  tasks: Task[];
  timezone: string;
  mutate: (tool: string, args: unknown, message: string) => Promise<unknown>;
  onClose: () => void;
  onSaved: () => void;
}) {
  useDialogFocus();
  const fresh = event.entity_id === "new",
    task = tasks.find((t) => t.id === event.task_id);
  const [record, setRecord] = useState<Record | null>(null),
    [connection, setConnection] = useState<GoogleStatus | null>(null);
  const [form, setForm] = useState<Fields>({
    title: task ? "Work on " + task.title : "",
    start: event.date + "T09:00",
    end: event.date + "T10:00",
    timezone,
    all_day: false,
    location: "",
    description: "",
    busy: true,
  });
  const [taskId, setTaskId] = useState(event.task_id ?? ""),
    [kind, setKind] = useState<"event" | "block">(
      event.kind === "block" ? "block" : "event",
    );
  const [calendar, setCalendar] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [deleting, setDeleting] = useState(false);
  const [compare, setCompare] = useState<{
    google: Fields & {
      edit_token: string | null;
      editable: boolean;
      read_only_reason: string;
    };
  }>();
  useEffect(() => {
    let live = true;
    api<GoogleStatus>("/integrations/google")
      .then((d) => {
        if (live) setConnection(d);
      })
      .catch(() => {});
    if (!fresh)
      api<Record>("/planning/" + event.entity_id)
        .then((d) => {
          if (!live) return;
          setRecord(d);
          setKind(d.kind);
          setTaskId(d.task_id ?? "");
          setForm({
            ...d.fields,
            start: d.fields.all_day
              ? d.fields.start
              : localDateTime(d.fields.start, d.fields.timezone),
            end: d.fields.all_day
              ? shiftDate(d.fields.end, -1)
              : localDateTime(d.fields.end, d.fields.timezone),
          });
        })
        .catch((e) => {
          if (live) setError(e.message);
        });
    return () => {
      live = false;
    };
  }, [event.entity_id, fresh]);
  const set = <K extends keyof Fields>(key: K, value: Fields[K]) =>
    setForm((f) => ({ ...f, [key]: value }));
  const pending =
    record &&
    ["queued", "running", "retrying"].includes(record.write_status ?? "");
  const review =
    record?.google_calendar_id &&
    !["synced", "local"].includes(record.google_state);
  async function act(tool: string, args: object) {
    setBusy(true);
    setError("");
    try {
      const saved = await mutate(
        tool,
        {
          ...(!fresh
            ? { entry_id: record!.id, expected_revision: record!.revision }
            : {}),
          ...args,
        },
        "Calendar entry saved",
      );
      if (saved) onSaved();
      else
        setError(
          "The request did not finish. Retry the same change to check its saved receipt.",
        );
      return saved;
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function save() {
    if (
      !form.title.trim() ||
      !form.start ||
      !form.end ||
      (!fresh && !record) ||
      pending ||
      review
    )
      return;
    return act(fresh ? "planning.create" : "planning.update", {
      ...form,
      end: form.all_day ? shiftDate(form.end, 1) : form.end,
      task_id: taskId || null,
      ...(fresh ? { kind, google_calendar_id: calendar || null } : {}),
    });
  }
  const baseForm = record
    ? {
        ...record.fields,
        start: record.fields.all_day
          ? record.fields.start
          : localDateTime(record.fields.start, record.fields.timezone),
        end: record.fields.all_day
          ? shiftDate(record.fields.end, -1)
          : localDateTime(record.fields.end, record.fields.timezone),
      }
    : {
        title: task ? "Work on " + task.title : "",
        start: event.date + "T09:00",
        end: event.date + "T10:00",
        timezone,
        all_day: false,
        location: "",
        description: "",
        busy: true,
      };
  const editorSchema = z.object({
    title: z.string().min(1).max(500),
    start: z.string().min(1).max(80),
    end: z.string().min(1).max(80),
    timezone: z.string().min(1).max(100),
    all_day: z.boolean(),
    location: z.string().max(1000),
    description: z.string().max(10000),
    busy: z.boolean(),
    task_id: nullableId(
      tasks.filter((t) => !t.archived && !t.is_template).map((t) => t.id),
    ),
    ...(fresh
      ? {
          kind: choice(["event", "block"]),
          google_calendar_id: nullableId(
            (connection?.calendars ?? [])
              .filter((c) => c.writable)
              .map((c) => c.id),
          ),
        }
      : {}),
  });
  useEditor({
    kind: "event",
    record_id: fresh ? null : event.entity_id,
    dirty:
      JSON.stringify(form) !== JSON.stringify(baseForm) ||
      taskId !== (record?.task_id ?? event.task_id ?? "") ||
      !!calendar ||
      (fresh && kind !== event.kind),
    busy: busy || (!fresh && !record) || !!pending || !!review,
    schema: editorSchema,
    values: {
      ...form,
      task_id: taskId || null,
      ...(fresh ? { kind, google_calendar_id: calendar || null } : {}),
    },
    save,
    close: onClose,
    patch: (v) => {
      const { task_id, kind: entryKind, google_calendar_id, ...fields } = v;
      setForm((f) => ({ ...f, ...fields }));
      if ("task_id" in v) setTaskId((task_id as string) || "");
      if ("kind" in v) setKind(entryKind as "event" | "block");
      if ("google_calendar_id" in v)
        setCalendar((google_calendar_id as string) || "");
    },
  });
  return (
    <div
      className="modal-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <section
        className="dialog google-event-editor"
        role="dialog"
        aria-modal="true"
        aria-labelledby="planning-title"
      >
        <div className="dialog-heading">
          <h2 id="planning-title">
            {fresh
              ? kind === "block"
                ? "Reserve task time"
                : "New event"
              : kind === "block"
                ? "Task work block"
                : "Event details"}
          </h2>
          <button
            type="button"
            aria-label="Close event"
            className="icon-button"
            onClick={onClose}
          >
            <X size={20} />
          </button>
        </div>
        {error && (
          <p role="alert" className="error-banner">
            {error}
          </p>
        )}
        {!fresh && !record ? (
          <p>Loading entry…</p>
        ) : (
          <>
            <p className="footnote">
              {record?.google_calendar_id
                ? "Google copy · " + record.google_state
                : fresh
                  ? "Kept in Eridani unless you publish a Google copy."
                  : "Saved in Eridani"}
              {record?.write_message ? " · " + record.write_message : ""}
            </p>
            {pending && (
              <p role="status">
                Waiting for Google to confirm. You can close this window and
                check again.
              </p>
            )}
            <form
              onSubmit={(e) => {
                e.preventDefault();
                void save();
              }}
            >
              <fieldset disabled={busy || !!pending || !!review}>
                {fresh && (
                  <label>
                    Entry type
                    <select
                      value={kind}
                      onChange={(e) => setKind(e.target.value as typeof kind)}
                    >
                      <option value="event">Appointment</option>
                      <option value="block">Task work block</option>
                    </select>
                  </label>
                )}
                <label>
                  Title
                  <input
                    aria-label="Event title"
                    required
                    maxLength={500}
                    value={form.title}
                    onChange={(e) => set("title", e.target.value)}
                  />
                </label>
                <label>
                  Linked task
                  <select
                    aria-label="Linked task"
                    required={kind === "block"}
                    value={taskId}
                    onChange={(e) => setTaskId(e.target.value)}
                  >
                    <option value="">No task</option>
                    {tasks
                      .filter((t) => !t.archived && !t.is_template)
                      .map((t) => (
                        <option key={t.id} value={t.id}>
                          {t.title}
                        </option>
                      ))}
                  </select>
                </label>
                <label className="event-checkbox">
                  <input
                    type="checkbox"
                    checked={form.all_day}
                    onChange={(e) => {
                      const v = e.target.checked;
                      setForm((f) => ({
                        ...f,
                        all_day: v,
                        start: v ? f.start.slice(0, 10) : f.start + "T09:00",
                        end: v ? f.end.slice(0, 10) : f.end + "T10:00",
                      }));
                    }}
                  />
                  All day
                </label>
                <div className="event-date-fields">
                  <label>
                    {form.all_day ? "First day" : "Starts"}
                    <input
                      aria-label="Event start"
                      required
                      type={form.all_day ? "date" : "datetime-local"}
                      value={form.start}
                      onChange={(e) => set("start", e.target.value)}
                    />
                  </label>
                  <label>
                    {form.all_day ? "Last day" : "Ends"}
                    <input
                      aria-label="Event end"
                      required
                      type={form.all_day ? "date" : "datetime-local"}
                      value={form.end}
                      onChange={(e) => set("end", e.target.value)}
                    />
                  </label>
                </div>
                <label>
                  Time zone
                  <input
                    aria-label="Event time zone"
                    required
                    value={form.timezone}
                    onChange={(e) => set("timezone", e.target.value)}
                  />
                </label>
                <label>
                  Location
                  <input
                    aria-label="Event location"
                    maxLength={1000}
                    value={form.location}
                    onChange={(e) => set("location", e.target.value)}
                  />
                </label>
                <label>
                  Notes
                  <textarea
                    aria-label="Event notes"
                    rows={4}
                    maxLength={10000}
                    value={form.description}
                    onChange={(e) => set("description", e.target.value)}
                  />
                </label>
                <label className="event-checkbox">
                  <input
                    type="checkbox"
                    checked={form.busy}
                    onChange={(e) => set("busy", e.target.checked)}
                  />
                  Reserve this time as busy
                </label>
                {fresh && (
                  <label>
                    Google copy
                    <select
                      aria-label="Google copy"
                      value={calendar}
                      onChange={(e) => setCalendar(e.target.value)}
                    >
                      <option value="">Keep in Eridani only</option>
                      {connection?.calendar_write_enabled &&
                        connection.calendars
                          .filter((c) => c.writable)
                          .map((c) => (
                            <option key={c.id} value={c.id}>
                              {c.title}
                            </option>
                          ))}
                    </select>
                  </label>
                )}
                <p className="footnote">
                  A work block reserves time. It does not change the task’s
                  deadline or alerts.
                </p>
                <button className="primary" type="submit">
                  Save event
                </button>
              </fieldset>
            </form>
            {!fresh && record && !pending && (
              <div className="planning-actions">
                {!record.google_calendar_id &&
                  connection?.calendar_write_enabled && (
                    <div>
                      <label>
                        Publish to Google
                        <select
                          aria-label="Publish calendar"
                          value={calendar}
                          onChange={(e) => setCalendar(e.target.value)}
                        >
                          <option value="">Choose calendar</option>
                          {connection.calendars
                            .filter((c) => c.writable)
                            .map((c) => (
                              <option key={c.id} value={c.id}>
                                {c.title}
                              </option>
                            ))}
                        </select>
                      </label>
                      <button
                        disabled={busy || !calendar}
                        onClick={() =>
                          void act("planning.publish", {
                            calendar_id: calendar,
                          })
                        }
                      >
                        Publish copy
                      </button>
                    </div>
                  )}
                {record.google_calendar_id && (
                  <>
                    <button
                      disabled={busy}
                      onClick={async () => {
                        setBusy(true);
                        try {
                          setCompare(
                            await api("/planning/" + record.id + "/comparison"),
                          );
                        } catch (e) {
                          setError((e as Error).message);
                        } finally {
                          setBusy(false);
                        }
                      }}
                    >
                      Compare Google copy
                    </button>
                    {compare && (
                      <div className="sync-comparison">
                        <h3>Google copy</h3>
                        <strong>{compare.google.title}</strong>
                        <p>
                          {compare.google.start} – {compare.google.end} ·{" "}
                          {compare.google.timezone}
                        </p>
                        <p className="plain-details">
                          {compare.google.description}
                        </p>
                        {compare.google.edit_token ? (
                          <>
                            <button
                              disabled={busy}
                              onClick={() =>
                                void act("planning.resolve", {
                                  choice: "google",
                                  edit_token: compare.google.edit_token,
                                })
                              }
                            >
                              Use Google version
                            </button>
                            <button
                              disabled={busy}
                              onClick={() =>
                                void act("planning.resolve", {
                                  choice: "eridani",
                                  edit_token: compare.google.edit_token,
                                })
                              }
                            >
                              Keep Eridani version
                            </button>
                          </>
                        ) : (
                          <p>{compare.google.read_only_reason}</p>
                        )}
                      </div>
                    )}
                    <button
                      disabled={busy}
                      onClick={() => void act("planning.unlink", {})}
                    >
                      Unlink · keep both copies
                    </button>
                  </>
                )}
                {!deleting ? (
                  <button
                    className="text-button"
                    disabled={busy || !!review}
                    onClick={() => setDeleting(true)}
                  >
                    Cancel event
                  </button>
                ) : (
                  <p>
                    Cancel this entry
                    {record.google_calendar_id
                      ? " and delete its Google copy"
                      : ""}
                    ? The task remains.
                    <button
                      disabled={busy}
                      onClick={() => void act("planning.delete", {})}
                    >
                      Confirm cancellation
                    </button>
                    <button onClick={() => setDeleting(false)}>
                      Keep event
                    </button>
                  </p>
                )}
              </div>
            )}
          </>
        )}
      </section>
    </div>
  );
}
