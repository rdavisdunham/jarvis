import { z } from "zod";
import { useEditor, choice } from "./editor-control";
import { useEffect, useRef, useState } from "react";
import { ExternalLink, X } from "lucide-react";
import { useDialogFocus } from "./components";
import { api, post } from "./api";
import { localDateTime, shiftDate } from "./workspace";
import type { CalendarEntry } from "./types";
import type { GoogleStatus } from "./GoogleSettings";

type EventFields = {
  title: string;
  start: string;
  end: string;
  all_day: boolean;
  timezone: string;
  location: string;
  description: string;
  busy: boolean;
};
type Detail = EventFields & {
  editable: boolean;
  edit_token: string | null;
  read_only_reason: string;
  calendar_title: string;
  recurring: boolean;
  scope: string;
  meeting_url?: string;
  organizer?: { displayName?: string; email?: string };
  attendees?: {
    displayName?: string;
    email?: string;
    responseStatus?: string;
  }[];
  attendees_omitted?: boolean;
  attachments?: { title: string; url: string }[];
};
type Write = { job_id: string; status: string; result?: { message?: string } };
const terminal = (status: string) =>
  ["succeeded", "failed", "cancelled", "unconfirmed"].includes(status);
export function GoogleEventDialog({
  event,
  timezone,
  onClose,
  onSettings,
  onSaved,
  mutate,
}: {
  event: CalendarEntry;
  timezone: string;
  onClose: () => void;
  onSettings: () => void;
  onSaved: () => void;
  mutate: (tool: string, args: unknown, message: string) => Promise<unknown>;
}) {
  useDialogFocus();
  const fresh = event.entity_id === "new";
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  const [connection, setConnection] = useState<GoogleStatus | null>(null);
  const [cached, setCached] = useState<Partial<Detail> | null>(null);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [calendar, setCalendar] = useState("");
  const [scope, setScope] = useState(event.recurring ? "occurrence" : "event");
  const [editing, setEditing] = useState(fresh);
  const [deleting, setDeleting] = useState(false);
  const [repeat, setRepeat] = useState("none");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [job, setJob] = useState<Write | null>(null);
  const [refresh, setRefresh] = useState(0);
  const [form, setForm] = useState<EventFields>({
    title: "",
    start: event.date + "T09:00",
    end: event.date + "T10:00",
    all_day: false,
    timezone,
    location: "",
    description: "",
    busy: true,
  });
  useEffect(() => {
    let current = true;
    api<GoogleStatus>("/integrations/google")
      .then((data) => {
        if (!current) return;
        setConnection(data);
        const choices = data.calendars.filter((c) => c.writable);
        setCalendar((id) =>
          choices.some((c) => c.id === id)
            ? id
            : (choices.find((c) => c.primary)?.id ?? choices[0]?.id ?? ""),
        );
      })
      .catch((e) => {
        if (current) setError(e.message);
      });
    return () => {
      current = false;
    };
  }, []);
  useEffect(() => {
    if (fresh) return;
    let current = true;
    setDetail(null);
    setError("");
    setEditing(false);
    setDeleting(false);
    setJob(null);
    api<Partial<Detail>>("/calendar/events/" + event.entity_id)
      .then((d) => {
        if (current) setCached(d);
      })
      .catch(() => {});
    post<Detail>("/calendar/event-detail", {
      event_id: event.entity_id,
      scope,
      ...(event.occurrence_start
        ? { occurrence_start: event.occurrence_start }
        : {}),
    })
      .then((data) => {
        if (!current) return;
        setDetail(data);
        setForm({
          ...data,
          start: data.all_day
            ? data.start
            : localDateTime(data.start, data.timezone),
          end: data.all_day
            ? shiftDate(data.end, -1)
            : localDateTime(data.end, data.timezone),
        });
      })
      .catch((e) => {
        if (current) setError(e.message);
      });
    return () => {
      current = false;
    };
  }, [fresh, event.entity_id, event.occurrence_start, scope, refresh]);
  const shown = detail ?? cached;
  const safeLink = (url: string) => /^https?:\/\//i.test(url);
  const set = <K extends keyof EventFields>(key: K, value: EventFields[K]) =>
    setForm((f) => ({ ...f, [key]: value }));
  const fields = () => ({
    title: form.title,
    start: form.start,
    end: form.all_day ? shiftDate(form.end, 1) : form.end,
    all_day: form.all_day,
    timezone: form.timezone,
    location: form.location,
    description: form.description,
    busy: form.busy,
  });
  async function checkJob(id: string) {
    for (let i = 0; i < 90 && alive.current; i++) {
      const status = await api<Write>("/calendar/writes/" + id);
      if (!alive.current) return;
      setJob(status);
      if (terminal(status.status)) {
        if (status.status === "succeeded") onSaved();
        else
          setError(
            status.result?.message ??
              "Google could not save this change. Reopen the event before trying again.",
          );
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }
  async function submit(remove = false, waitForRemote = true) {
    setBusy(true);
    setError("");
    try {
      if (job && !terminal(job.status)) {
        await checkJob(job.job_id);
        return;
      }
      const tool = remove
        ? "calendar.delete"
        : fresh
          ? "calendar.create"
          : "calendar.update";
      const args = remove
        ? { edit_token: detail?.edit_token }
        : fresh
          ? { ...fields(), calendar_id: calendar, repeat }
          : { ...fields(), edit_token: detail?.edit_token };
      const accepted = (await mutate(tool, args, "Calendar change queued")) as
        | Write
        | undefined;
      if (!accepted) {
        setError(
          "The request did not finish. Retry the same change to check its saved receipt.",
        );
        return;
      }
      setJob(accepted);
      window.dispatchEvent(new Event("eri-calendar-write"));
      if (waitForRemote) await checkJob(accepted.job_id);
      else
        void checkJob(accepted.job_id).catch((e) => {
          if (alive.current) setError((e as Error).message);
        });
      return {
        outcome: "queued",
        saved: false,
        job_id: accepted.job_id,
        status: accepted.status,
        note: "Google has accepted a job. Use calendar_write_status to confirm the remote result.",
      };
    } catch (e) {
      if (alive.current) setError((e as Error).message);
    } finally {
      if (alive.current) setBusy(false);
    }
  }
  const canCreate = !!connection?.calendar_write_enabled && !!calendar;
  const canEdit = fresh ? canCreate : !!detail?.editable;
  const pending = job && !terminal(job.status);
  const blocked = !!job && terminal(job.status) && job.status !== "succeeded";
  const baseForm = detail
    ? {
        title: detail.title,
        start: detail.all_day
          ? detail.start
          : localDateTime(detail.start, detail.timezone),
        end: detail.all_day
          ? shiftDate(detail.end, -1)
          : localDateTime(detail.end, detail.timezone),
        all_day: detail.all_day,
        timezone: detail.timezone,
        location: detail.location,
        description: detail.description,
        busy: detail.busy,
      }
    : {
        title: "",
        start: event.date + "T09:00",
        end: event.date + "T10:00",
        all_day: false,
        timezone,
        location: "",
        description: "",
        busy: true,
      };
  const comparable = Object.fromEntries(
    Object.keys(baseForm).map((k) => [k, form[k as keyof EventFields]]),
  );
  const schema = z.object({
    title: z.string().min(1).max(500),
    start: z.string().min(1).max(80),
    end: z.string().min(1).max(80),
    all_day: z.boolean(),
    timezone: z.string().min(1).max(100),
    location: z.string().max(1000),
    description: z.string().max(10000),
    busy: z.boolean(),
    ...(fresh
      ? {
          calendar_id: choice(
            (connection?.calendars ?? [])
              .filter((c) => c.writable)
              .map((c) => c.id),
          ),
          repeat: choice(["none", "daily", "weekdays", "weekly", "monthly"]),
        }
      : {}),
  });
  useEditor({
    kind: "google_event",
    record_id: fresh ? null : event.entity_id,
    dirty:
      JSON.stringify(comparable) !== JSON.stringify(baseForm) ||
      repeat !== "none",
    busy: busy || !!pending || !!blocked || !canEdit,
    schema,
    values: {
      ...comparable,
      ...(fresh ? { calendar_id: calendar, repeat } : {}),
      scope,
    },
    patch: (v) => {
      if (!canEdit)
        throw new Error(
          detail?.read_only_reason || "Calendar writes are unavailable.",
        );
      setEditing(true);
      const { calendar_id, repeat: nextRepeat, ...fields } = v;
      setForm((f) => ({ ...f, ...fields }));
      if ("calendar_id" in v) setCalendar(calendar_id as string);
      if ("repeat" in v) setRepeat(nextRepeat as string);
    },
    save: async () => {
      if (!canEdit || !form.title.trim()) return;
      return submit(false, false);
    },
    close: onClose,
  });
  const fmt = (value: string) =>
    new Intl.DateTimeFormat(undefined, {
      timeZone: timezone,
      dateStyle: "medium",
      timeStyle: "short",
    }).format(new Date(value));
  return (
    <div className="modal-backdrop">
      <section
        className="dialog google-event-editor"
        role="dialog"
        aria-modal="true"
        aria-labelledby="google-event-title"
      >
        <div className="dialog-heading">
          <h2 id="google-event-title">
            {fresh
              ? "New calendar event"
              : editing
                ? "Edit calendar event"
                : event.title}
          </h2>
          <button
            className="icon-button"
            aria-label="Close calendar event"
            onClick={onClose}
          >
            <X size={20} />
          </button>
        </div>
        {!fresh && event.recurring && (
          <label className="calendar-scope">
            Apply to
            <select
              aria-label="Recurring event scope"
              value={scope}
              disabled={busy || !!pending}
              onChange={(e) => setScope(e.target.value)}
            >
              <option value="occurrence">This occurrence</option>
              <option value="series">Entire series</option>
            </select>
          </label>
        )}
        {scope === "series" && (
          <p className="integration-hint">
            Changes apply to the entire recurring series. Dates below refer to
            the series start.
          </p>
        )}
        {error && (
          <p className="error-banner" role="alert">
            {error}
          </p>
        )}
        {job && (
          <p className="calendar-write-feedback" role="status">
            {job.status === "succeeded"
              ? "Saved in Google Calendar."
              : job.status === "unconfirmed"
                ? "Outcome not confirmed. Check Google Calendar before creating another event."
                : terminal(job.status)
                  ? job.result?.message
                  : "Saving in Google Calendar… You can close this and check Recent calendar changes."}
          </p>
        )}
        {!fresh && !detail && !error && (
          <p role="status">Loading current event…</p>
        )}
        {fresh && connection && !canCreate && (
          <p className="integration-hint">
            {connection.calendar_write_enabled
              ? "Select a calendar you can edit in Settings."
              : "Enable Calendar editing in Settings to create events."}{" "}
            <button className="text-button" onClick={onSettings}>
              Open Settings
            </button>
          </p>
        )}
        {!fresh && detail && !detail.editable && (
          <p className="footnote">
            {detail.read_only_reason}{" "}
            {!connection?.calendar_write_enabled && (
              <button className="text-button" onClick={onSettings}>
                Open Settings
              </button>
            )}
          </p>
        )}
        {editing && canEdit ? (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void submit();
            }}
          >
            <fieldset disabled={busy || !!pending || blocked}>
              {fresh && (
                <label>
                  Calendar
                  <select
                    aria-label="Event calendar"
                    value={calendar}
                    onChange={(e) => setCalendar(e.target.value)}
                  >
                    {connection?.calendars
                      .filter((c) => c.writable)
                      .map((c) => (
                        <option key={c.id} value={c.id}>
                          {c.title}
                        </option>
                      ))}
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
              <label className="event-checkbox">
                <input
                  type="checkbox"
                  checked={form.all_day}
                  onChange={(e) => {
                    const value = e.target.checked;
                    setForm((f) => ({
                      ...f,
                      all_day: value,
                      start: value ? f.start.slice(0, 10) : f.start + "T09:00",
                      end: value ? f.end.slice(0, 10) : f.end + "T10:00",
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
              {fresh && (
                <label>
                  Repeat
                  <select
                    aria-label="Event repeat"
                    value={repeat}
                    onChange={(e) => setRepeat(e.target.value)}
                  >
                    <option value="none">Does not repeat</option>
                    <option value="daily">Every day</option>
                    <option value="weekly">Every week</option>
                    <option value="monthly">Every month</option>
                  </select>
                </label>
              )}
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
                  maxLength={10000}
                  rows={3}
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
                Blocks availability
              </label>
              <button className="primary" type="submit">
                {busy ? "Saving…" : fresh ? "Create event" : "Save event"}
              </button>
            </fieldset>
          </form>
        ) : (
          !fresh && (
            <>
              <p>
                {event.all_day
                  ? "All day · " + event.date
                  : event.at
                    ? fmt(event.at) +
                      (event.end_at ? " – " + fmt(event.end_at) : "")
                    : event.date}
              </p>
              <p className="footnote">
                {event.calendar_title} · {timezone}
              </p>
              {(shown?.location || event.location) && (
                <p>{shown?.location || event.location}</p>
              )}
              {(shown?.description || event.description) && (
                <p className="event-description">
                  {shown?.description || event.description}
                </p>
              )}
              {!detail && cached && (
                <p className="footnote">
                  Showing synced details while the current Google copy is
                  unavailable.
                </p>
              )}
              {shown?.meeting_url && safeLink(shown.meeting_url) && (
                <p>
                  <a href={shown.meeting_url} target="_blank" rel="noreferrer">
                    Join meeting
                  </a>
                </p>
              )}
              {shown?.organizer && (
                <p className="footnote">
                  Organizer:{" "}
                  {shown.organizer.displayName || shown.organizer.email}
                </p>
              )}
              {!!shown?.attendees?.length && (
                <details>
                  <summary>
                    Guests ({shown.attendees.length}
                    {shown.attendees_omitted ? "+" : ""})
                  </summary>
                  {shown.attendees.map((a, i) => (
                    <p key={i}>
                      {a.displayName || a.email} ·{" "}
                      {a.responseStatus ?? "No response"}
                    </p>
                  ))}
                </details>
              )}
              {shown?.attachments
                ?.filter((a) => safeLink(a.url))
                .map((a, i) => (
                  <p key={i}>
                    <a href={a.url} target="_blank" rel="noreferrer">
                      {a.title || "Attachment"}
                    </a>
                  </p>
                ))}
              {!event.busy && (
                <p className="footnote">This event is marked as free time.</p>
              )}
              {detail?.editable && !deleting && (
                <div className="event-editor-actions">
                  <button
                    className="primary"
                    disabled={busy || !!pending || blocked}
                    onClick={() => setEditing(true)}
                  >
                    Edit event
                  </button>
                  <button
                    className="text-button danger"
                    disabled={busy || !!pending || blocked}
                    onClick={() => setDeleting(true)}
                  >
                    Delete event
                  </button>
                </div>
              )}
            </>
          )
        )}
        {deleting && (
          <div className="disconnect-confirm">
            <p>
              Delete{" "}
              {scope === "series"
                ? "the entire series"
                : scope === "occurrence"
                  ? "this occurrence"
                  : "this event"}{" "}
              from Google Calendar?
            </p>
            <button
              className="secondary"
              disabled={busy || !!pending || blocked}
              onClick={() => void submit(true)}
            >
              Confirm delete
            </button>
            <button
              className="text-button"
              disabled={busy}
              onClick={() => setDeleting(false)}
            >
              Keep event
            </button>
          </div>
        )}
        {pending && !busy && (
          <button className="secondary" onClick={() => void submit()}>
            Check save status
          </button>
        )}
        {!fresh && !busy && (blocked || (error && !detail)) && (
          <button
            className="secondary"
            onClick={() => setRefresh((n) => n + 1)}
          >
            Reload event
          </button>
        )}
        {event.url && (
          <a
            className="secondary compact external-link"
            href={event.url}
            target="_blank"
            rel="noopener noreferrer"
          >
            Open in Google Calendar
            <ExternalLink size={15} />
          </a>
        )}
      </section>
    </div>
  );
}
