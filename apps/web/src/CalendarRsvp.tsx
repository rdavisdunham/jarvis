import { useEffect, useRef, useState } from "react";
import { api, post } from "./api";
import type { CalendarEntry } from "./types";

type Write = { job_id: string; status: string; result?: { message?: string; response?: string } };
type Phase = { kind: "idle" } | { kind: "pending"; job: string } | { kind: "saved"; job: string } | { kind: "error"; message: string };

export const RSVP_CHOICES = [["accepted", "Yes"], ["tentative", "Maybe"], ["declined", "No"]] as const;
export const responseLabel = (value?: string | null) =>
  ({ accepted: "Going", tentative: "Maybe", declined: "Not going", needsAction: "Not answered" })[value ?? ""] ?? "";
const terminal = (status: string) => ["succeeded", "failed", "cancelled", "unconfirmed"].includes(status);

/** Shown on a Google event row when the connected account is a guest (my_response is set) and can write. */
export const canAnswer = (e: CalendarEntry) => e.kind === "google" && !!e.my_response && !e.read_only;

/**
 * Yes / Maybe / No on a calendar row. Same path as the event dialog: a fresh read for a response token, a queued
 * calendar.respond, then the durable receipt. Pending, saved and conflict states stay on the row, with Revert.
 * The organizer is notified (sendUpdates=all), as when answering in Google Calendar.
 */
export function CalendarRsvp({
  entry,
  mutate,
  onChanged,
}: {
  entry: CalendarEntry;
  mutate: (tool: string, args: unknown, message: string) => Promise<unknown>;
  onChanged: () => void;
}) {
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [shown, setShown] = useState(entry.my_response);
  const alive = useRef(true);
  useEffect(() => () => void (alive.current = false), []);
  useEffect(() => setShown(entry.my_response), [entry.my_response]);

  async function follow(id: string): Promise<Write> {
    for (let i = 0; i < 90 && alive.current; i++) {
      const status = await api<Write>("/calendar/writes/" + id);
      if (terminal(status.status)) return status;
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
    return { job_id: id, status: "running" };
  }
  async function run(start: () => Promise<Write | undefined>, response?: string) {
    try {
      const accepted = await start();
      if (!accepted) throw new Error("The request did not finish. Check Recent calendar changes before retrying.");
      setPhase({ kind: "pending", job: accepted.job_id });
      window.dispatchEvent(new Event("eri-calendar-write"));
      const done = await follow(accepted.job_id);
      if (!alive.current) return;
      if (done.status === "succeeded") {
        setShown(done.result?.response ?? response ?? shown);
        setPhase(response ? { kind: "saved", job: accepted.job_id } : { kind: "idle" });
        onChanged();
      } else if (terminal(done.status)) {
        setPhase({ kind: "error", message: done.result?.message ?? "Google could not save your response. Open the event to check it." });
      }
    } catch (e) {
      if (alive.current) setPhase({ kind: "error", message: (e as Error).message });
    }
  }
  const answer = (response: string) =>
    run(async () => {
      setPhase({ kind: "pending", job: "" });
      const detail = await post<{ response_token?: string | null; read_only_reason?: string }>("/calendar/event-detail", {
        event_id: entry.entity_id,
        scope: entry.recurring ? "occurrence" : "event",
        ...(entry.recurring && entry.occurrence_start ? { occurrence_start: entry.occurrence_start } : {}),
      });
      if (!detail.response_token)
        throw new Error("This invitation can't be answered here. Open it in Google Calendar.");
      return (await mutate("calendar.respond", { response_token: detail.response_token, response, notify: true },
        "Response queued")) as Write | undefined;
    }, response);
  const revert = (job: string) =>
    run(async () => (await mutate("calendar.response_revert", { job_id: job }, "Response reversal queued")) as Write | undefined);

  const pending = phase.kind === "pending";
  return (
    <div className="cal-rsvp" role="group" aria-label={"Your response to " + entry.title}>
      <span className="cal-rsvp-label">{responseLabel(shown)}</span>
      {RSVP_CHOICES.map(([value, text]) => (
        <button key={value} type="button" className="btn btn-sm btn-secondary" aria-pressed={shown === value}
          disabled={pending || shown === value} onClick={() => void answer(value)}>{text}</button>
      ))}
      <span className="cal-rsvp-state" role="status">
        {pending ? "Sending to Google…" : phase.kind === "saved" ? "Saved in Google." : ""}
        {phase.kind === "error" && <span className="cal-rsvp-error">{phase.message}</span>}
      </span>
      {phase.kind === "saved" && (
        <button type="button" className="text-button" onClick={() => void revert(phase.job)}>Revert</button>
      )}
    </div>
  );
}
