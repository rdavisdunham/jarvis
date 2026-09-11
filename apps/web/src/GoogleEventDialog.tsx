import { ExternalLink, X } from "lucide-react";
import { useDialogFocus } from "./components";
import type { CalendarEntry } from "./types";
export function GoogleEventDialog({
  event,
  timezone,
  onClose,
}: {
  event: CalendarEntry;
  timezone: string;
  onClose: () => void;
}) {
  useDialogFocus();
  const fmt = (v: string) =>
    new Intl.DateTimeFormat(undefined, {
      timeZone: timezone,
      dateStyle: "medium",
      timeStyle: "short",
    }).format(new Date(v));
  return (
    <div className="modal-backdrop">
      <section
        className="dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="google-event-title"
      >
        <div className="dialog-heading">
          <h2 id="google-event-title">{event.title}</h2>
          <button
            className="icon-button"
            aria-label="Close calendar event"
            onClick={onClose}
          >
            <X size={20} />
          </button>
        </div>
        <p>
          {event.all_day
            ? "All day · " + event.date
            : event.at
              ? fmt(event.at) + (event.end_at ? " – " + fmt(event.end_at) : "")
              : event.date}
        </p>
        <p className="footnote">
          {event.calendar_title} · {timezone} · Read-only
        </p>
        {event.location && <p>{event.location}</p>}
        {!event.busy && (
          <p className="footnote">This event is marked as free time.</p>
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
