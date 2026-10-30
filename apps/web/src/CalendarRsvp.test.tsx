import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { CalendarRsvp, canAnswer, responseLabel } from "./CalendarRsvp";
import { CalendarView } from "./CalendarView";
import { ReconnectCallout, grantSummary, reconnectPurpose, type GoogleStatus } from "./GoogleSettings";
import type { CalendarEntry } from "./types";

const invite = (changes: Partial<CalendarEntry> = {}): CalendarEntry => ({
  id: "google:e1:2026-09-18T15:00:00-05:00:2026-09-18", entity_id: "e1", kind: "google", title: "Design review",
  date: "2026-09-18", at: "2026-09-18T15:00:00-05:00", end_at: "2026-09-18T16:00:00-05:00", status: "active",
  project_id: null, task_id: null, revision: 1, projected: false, notification_id: null, busy: true,
  read_only: false, my_response: "needsAction", declined: false, ...changes,
});
const mutate = async () => undefined;
const view = (events: CalendarEntry[], withMutate = true) => renderToStaticMarkup(
  <CalendarView day="2026-09-18" today="2026-09-18" zone="America/Chicago" mode="day" days={["2026-09-18"]}
    events={events} loaded truncated={false} error="" filtered={false} onDay={() => {}} onMode={() => {}}
    onEntry={() => {}} onRetry={() => {}} {...(withMutate ? { mutate } : {})} />,
);

describe("calendar RSVP", () => {
  it("offers Yes / Maybe / No only on writable invitations", () => {
    expect(canAnswer(invite())).toBe(true);
    expect(canAnswer(invite({ read_only: true }))).toBe(false);
    expect(canAnswer(invite({ my_response: null }))).toBe(false);  // Own events and events without guests.
    expect(canAnswer(invite({ kind: "event" }))).toBe(false);
    const html = renderToStaticMarkup(<CalendarRsvp entry={invite({ my_response: "tentative" })} mutate={mutate} onChanged={() => {}} />);
    expect(html).toContain("Yes");
    expect(html).toContain("No");
    expect(html).toContain("Maybe");
    expect(html).toMatch(/aria-pressed="true"[^>]*disabled=""[^>]*>Maybe</);
    expect(html).toContain('role="status"');
    expect(responseLabel("declined")).toBe("Not going");
  });

  it("puts the answer beside the row, never inside its button, and dims declined events", () => {
    const html = view([invite(), invite({ id: "x", entity_id: "e2", title: "Skipped", declined: true, busy: false, my_response: "declined" })]);
    expect(html.match(/class="cal-rsvp"/g)).toHaveLength(2);
    expect(html).not.toMatch(/<button[^>]*agenda-row[^>]*>(?:(?!<\/button>).)*cal-rsvp/s);
    expect(html).toMatch(/agenda-row cal-kind-google is-declined/);
    const readOnly = view([invite({ declined: true, my_response: "declined" })], false);
    expect(readOnly).not.toContain("cal-rsvp");
    expect(readOnly).toContain(">Not going<");
  });
});

describe("Google reconnect", () => {
  const status = (changes: Partial<GoogleStatus>): GoogleStatus => ({
    configured: true, linked: true, email: "owner@example.test", calendar_enabled: true,
    calendar_write_enabled: true, status: "ready", syncing: false, error: "", last_sync_at: null, stale: false,
    calendars: [], ...changes,
  });
  it("explains a revoked grant and reconnects with the same permissions", () => {
    expect(renderToStaticMarkup(<ReconnectCallout data={status({})} disabled={false} onReconnect={() => {}} />)).toBe("");
    const grant = { refresh_token_issued_at: null, refresh_token_age_days: 7.1, last_refresh_at: null,
      rejected_at: "2026-10-30T00:00:00Z", testing_mode: "likely" as const,
      testing_mode_reason: "The last grant stopped working after 7.0 days." };
    const html = renderToStaticMarkup(<ReconnectCallout data={status({ status: "needs_reconnect", error: "revoked", grant })}
      disabled={false} onReconnect={() => {}} />);
    expect(html).toContain("stopped accepting");
    expect(html).toContain("7.0 days");
    expect(html).toContain("Reconnect Google Calendar");
    expect(reconnectPurpose(status({}))).toBe("calendar_write");
    expect(reconnectPurpose(status({ calendar_write_enabled: false }))).toBe("calendar");
    expect(grantSummary(grant)).toMatch(/^Access granted 7\.1 days ago, not refreshed yet\.$/);
  });
});
