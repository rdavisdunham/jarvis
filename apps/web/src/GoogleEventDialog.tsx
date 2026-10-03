import { SourceDetails, SourceNotes, useInlineGuard } from "./SourceDetails";
import { Dialog } from "./ux";
import { z } from "zod";
import { useEditor, choice } from "./editor-control";
import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  CalendarDays,
  ExternalLink,
  Paperclip,
  Pencil,
  Video,
  X,
} from "lucide-react";
import "./calendar.css";
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
  onClose: closeParent,
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
  const annotation=useInlineGuard();
  const onClose=()=>void annotation.flush().then(closeParent).catch(()=>{});
  const fresh = event.entity_id === "new";
  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  const [connection, setConnection] = useState<GoogleStatus | null>(null);
  const [cached, setCached] = useState<Partial<
    Omit<Detail, "start" | "end" | "all_day" | "busy">
  > | null>(null);
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
    api<Partial<Omit<Detail, "start" | "end" | "all_day" | "busy">>>(
      "/calendar/events/" + event.entity_id,
    )
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
    dirty: annotation.dirty ||
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
    beforeLeave: annotation.flush,
    close: onClose,
  });
  const fmt = (value: string) =>
    new Intl.DateTimeFormat(undefined, {
      timeZone: timezone,
      dateStyle: "medium",
      timeStyle: "short",
    }).format(new Date(value));
  const when =
    (detail?.all_day ?? event.all_day)
      ? (detail?.start ?? event.date) +
        (detail?.end ? " – " + shiftDate(detail.end, -1) : "")
      : (detail?.start ?? event.at)
        ? fmt((detail?.start ?? event.at)!) +
          ((detail?.end ?? event.end_at)
            ? " – " + fmt((detail?.end ?? event.end_at)!)
            : "")
        : event.date;
  const locked = busy || !!pending || blocked;
  const viewProps: [string, ReactNode][] = [
    ["Calendar", event.calendar_title || "Google Calendar"],
    ["Time zone", timezone],
    ...(shown?.organizer
      ? ([
          [
            "Organizer",
            shown.organizer.displayName || shown.organizer.email,
          ],
        ] as [string, ReactNode][])
      : []),
    [
      "Availability",
      (detail?.busy ?? event.busy) === false ? "Free" : "Busy",
    ],
  ];
  const footer =
    editing && canEdit ? (
      <div className="dialog-actions">
        {event.url && (
          <a
            className="btn btn-ghost external-link dialog-actions-start"
            href={event.url}
            target="_blank"
            rel="noopener noreferrer"
          >
            Open in Google Calendar
            <ExternalLink size={15} aria-hidden="true" />
          </a>
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
        <button
          className="primary"
          type="submit"
          form="google-event-form"
          disabled={locked}
        >
          {busy ? "Saving…" : fresh ? "Create event" : "Save event"}
        </button>
      </div>
    ) : (
      !fresh &&
      (event.url ||
        pending ||
        blocked ||
        (error && !detail) ||
        (detail?.editable && !deleting)) && (
        <div className="dialog-actions">
          {detail?.editable && !deleting && (
            <button
              className="btn btn-danger dialog-actions-start"
              disabled={locked}
              onClick={() => setDeleting(true)}
            >
              Delete event
            </button>
          )}
          {pending && !busy && (
            <button className="secondary" onClick={() => void submit()}>
              Check save status
            </button>
          )}
          {!busy && (blocked || (error && !detail)) && (
            <button
              className="secondary"
              onClick={() => setRefresh((n) => n + 1)}
            >
              Reload event
            </button>
          )}
          {event.url && (
            <a
              className="secondary external-link"
              href={event.url}
              target="_blank"
              rel="noopener noreferrer"
            >
              Open in Google Calendar
              <ExternalLink size={15} aria-hidden="true" />
            </a>
          )}
        </div>
      )
    );
  return (
    <Dialog
      className="dialog cal-dialog google-event-editor"
      aria-labelledby="google-event-title"
    >
      <div className="dialog-heading">
        <div className="cal-dialog-heading-text">
          {!editing && (
            <span className="chip-row">
              <span className="chip calendar-type-badge cal-kind-google">
                <CalendarDays size={13} aria-hidden="true" />
                Google event
              </span>
              <span className="chip">No task completion</span>
            </span>
          )}
          <h2 id="google-event-title">
            {fresh
              ? "New calendar event"
              : editing
                ? "Edit calendar event"
                : (shown?.title ?? event.title)}
          </h2>
        </div>
        <div className="cal-dialog-heading-actions">
          {!fresh && !editing && (
            <button
              className="btn btn-ghost btn-sm detail-edit"
              aria-label="Edit event"
              title={detail?.read_only_reason || "Edit event"}
              disabled={!canEdit || busy || !!pending || blocked}
              onClick={() => setEditing(true)}
            >
              <Pencil size={15} aria-hidden="true" />
              <span>Edit</span>
            </button>
          )}
          <button
            className="btn-icon"
            aria-label="Close calendar event"
            onClick={onClose}
          >
            <X size={20} />
          </button>
        </div>
      </div>
      {!fresh && event.recurring && (
        <div className="cal-form-grid cal-scope-row">
          <label className="field calendar-scope">
            <span className="field-label-text">Apply to</span>
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
        </div>
      )}
      {scope === "series" && (
        <p className="cal-notice">
          Changes apply to the entire recurring series. Dates below refer to the
          series start.
        </p>
      )}
      {error && (
        <p className="error-banner" role="alert">
          {error}
        </p>
      )}
      {job && (
        <p
          className={
            "cal-notice calendar-write-feedback" +
            (job.status === "succeeded" ? " cal-notice-success" : "")
          }
          role="status"
        >
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
        <p role="status" className="cal-loading-line">
          Loading current event…
        </p>
      )}
      {fresh && connection && !canCreate && (
        <p className="cal-notice">
          {connection.calendar_write_enabled
            ? "Select a calendar you can edit in Settings."
            : "Enable Calendar editing in Settings to create events."}{" "}
          <button className="text-button" onClick={onSettings}>
            Open Settings
          </button>
        </p>
      )}
      {!fresh && detail && !detail.editable && (
        <p className="cal-notice">
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
          id="google-event-form"
          onSubmit={(e) => {
            e.preventDefault();
            void submit();
          }}
        >
          <fieldset className="cal-form cal-form-grid" disabled={locked}>
            {fresh && (
              <label className="field span-2">
                <span className="field-label-text">Calendar</span>
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
            <label className="field span-2">
              <span className="field-label-text">Title</span>
              <input
                aria-label="Event title"
                required
                maxLength={500}
                value={form.title}
                onChange={(e) => set("title", e.target.value)}
              />
            </label>
            <label className="field">
              <span className="field-label-text">
                {form.all_day ? "First day" : "Starts"}
              </span>
              <input
                aria-label="Event start"
                required
                type={form.all_day ? "date" : "datetime-local"}
                value={form.start}
                onChange={(e) => set("start", e.target.value)}
              />
            </label>
            <label className="field">
              <span className="field-label-text">
                {form.all_day ? "Last day" : "Ends"}
              </span>
              <input
                aria-label="Event end"
                required
                type={form.all_day ? "date" : "datetime-local"}
                value={form.end}
                onChange={(e) => set("end", e.target.value)}
              />
            </label>
            <label className="field">
              <span className="field-label-text">Time zone</span>
              <input
                aria-label="Event time zone"
                required
                value={form.timezone}
                onChange={(e) => set("timezone", e.target.value)}
              />
            </label>
            {fresh ? (
              <label className="field">
                <span className="field-label-text">Repeat</span>
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
            ) : (
              <span aria-hidden="true" />
            )}
            <div className="cal-checks span-2">
              <label className="cal-check event-checkbox">
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
              <label className="cal-check event-checkbox">
                <input
                  type="checkbox"
                  checked={form.busy}
                  onChange={(e) => set("busy", e.target.checked)}
                />
                Blocks availability
              </label>
            </div>
            <label className="field span-2">
              <span className="field-label-text">Location</span>
              <input
                aria-label="Event location"
                maxLength={1000}
                value={form.location}
                onChange={(e) => set("location", e.target.value)}
              />
            </label>
            <label className="field span-2">
              <span className="field-label-text">Notes</span>
              <textarea
                aria-label="Event notes"
                maxLength={10000}
                rows={3}
                value={form.description}
                onChange={(e) => set("description", e.target.value)}
              />
            </label>
          </fieldset>
        </form>
      ) : (
        !fresh && (
          <div className="cal-detail-layout">
                        <div className="cal-detail-main">
              <SourceDetails source={event.source}/><SourceNotes kind="google" id={event.entity_id} onGuard={annotation.update}/>
              <div className="cal-when">
                <CalendarDays size={18} aria-hidden="true" />
                <div>
                  {when}
                  {(detail?.all_day ?? event.all_day) && <small>All day</small>}
                </div>
              </div>
              {(shown?.location ?? event.location) && (
                <section className="cal-section">
                  <h3>Location</h3>
                  <p>{shown?.location ?? event.location}</p>
                </section>
              )}
              {(shown?.description ?? event.description) && (
                <section className="cal-section">
                  <h3>Details</h3>
                  <p className="cal-prose event-description">
                    {shown?.description ?? event.description}
                  </p>
                </section>
              )}
              {!detail && cached && (
                <p className="cal-form-hint">
                  Showing synced details while the current Google copy is
                  unavailable.
                </p>
              )}
              {shown?.meeting_url && safeLink(shown.meeting_url) && (
                <p>
                  <a
                    className="btn btn-soft"
                    href={shown.meeting_url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    <Video size={16} aria-hidden="true" />
                    Join meeting
                  </a>
                </p>
              )}
              {!!shown?.attendees?.length && (
                <section className="cal-section">
                  <h3>
                    Guests ({shown.attendees.length}
                    {shown.attendees_omitted ? "+" : ""})
                  </h3>
                  <ul className="cal-guests">
                    {shown.attendees.map((a, i) => (
                      <li key={i}>
                        <span>{a.displayName || a.email}</span>
                        <span>{guestStatus(a.responseStatus)}</span>
                      </li>
                    ))}
                  </ul>
                </section>
              )}
              {!!shown?.attachments?.filter((a) => safeLink(a.url)).length && (
                <section className="cal-section">
                  <h3>Attachments</h3>
                  <div className="cal-links">
                    {shown
                      .attachments!.filter((a) => safeLink(a.url))
                      .map((a, i) => (
                        <a key={i} href={a.url} target="_blank" rel="noreferrer">
                          <Paperclip size={15} aria-hidden="true" />
                          {a.title || "Attachment"}
                        </a>
                      ))}
                  </div>
                </section>
              )}
            </div>
            <aside className="cal-aside" aria-label="Properties">
              <dl className="cal-props">
                {viewProps.map(([term, value]) => (
                  <div key={term}>
                    <dt>{term}</dt>
                    <dd>{value}</dd>
                  </div>
                ))}
              </dl>
            </aside>
          </div>
        )
      )}
      {deleting && (
        <div className="cal-confirm disconnect-confirm">
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
            className="btn btn-ghost"
            disabled={busy}
            onClick={() => setDeleting(false)}
          >
            Keep event
          </button>
          <button
            className="btn btn-danger"
            disabled={locked}
            onClick={() => void submit(true)}
          >
            Confirm delete
          </button>
        </div>
      )}
      {footer}
    </Dialog>
  );
}
function guestStatus(status?: string) {
  if (status === "accepted") return "Accepted";
  if (status === "declined") return "Declined";
  if (status === "tentative") return "Maybe";
  return "No response";
}
