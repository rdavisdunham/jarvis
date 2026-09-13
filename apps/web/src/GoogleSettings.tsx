import { useEffect, useState } from "react";
import { Check, ExternalLink, RefreshCw } from "lucide-react";
import { api, post } from "./api";

export interface GoogleStatus {
  configured: boolean;
  linked: boolean;
  email: string | null;
  calendar_enabled: boolean;
  calendar_write_enabled: boolean;
  status: string;
  syncing: boolean;
  error: string;
  last_sync_at: string | null;
  stale: boolean;
  calendars: {
    id: string;
    title: string;
    timezone: string;
    selected: boolean;
    available: boolean;
    primary: boolean;
    revision: number;
    access_role: string;
    writable: boolean;
    last_sync_at: string | null;
  }[];
}
export async function startGoogle(
  purpose: "login" | "link" | "calendar" | "calendar_write",
) {
  const result = await post<{ url: string }>("/auth/google/start", { purpose });
  location.assign(result.url);
}
export function GoogleSettings({
  revision,
  voiceActive,
  mutate,
}: {
  revision: number;
  voiceActive: boolean;
  mutate: (tool: string, args: unknown, message: string) => Promise<unknown>;
}) {
  const [data, setData] = useState<GoogleStatus | null>(null);
  const [error, setError] = useState(""),
    [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false),
    [refresh, setRefresh] = useState(0);
  const [disconnecting, setDisconnecting] = useState(false);
  const [unlinking, setUnlinking] = useState(false);
  useEffect(() => {
    let active = true;
    api<GoogleStatus>("/integrations/google")
      .then((d) => {
        if (active) setData(d);
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, [revision, refresh]);
  async function act(fn: () => Promise<unknown>) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRefresh((n) => n + 1);
      setBusy(false);
    }
  }
  return (
    <section className="google-settings">
      <h2>Google</h2>
      <p>
        Sign in with your account and bring your calendars into Eri’s view of
        your day.
      </p>
      {error && (
        <p className="error-banner" role="alert">
          {error}
          <button
            onClick={() => {
              setError("");
              setRefresh((n) => n + 1);
            }}
          >
            Retry
          </button>
        </p>
      )}
      {notice && <p role="status">{notice}</p>}
      {!data ? (
        <p role="status">Loading connection…</p>
      ) : (
        <>
          {!data.configured && (
            <p className="integration-hint">
              Google connection needs setup on your home server. Pairing still
              works.
            </p>
          )}
          <div className="setting-row">
            <span>
              <strong>Sign in with Google</strong>
              <small>
                {data.linked
                  ? data.email
                  : "Link your account from this paired device."}
              </small>
            </span>
            {data.linked ? (
              <span className="connection-label">
                <Check size={16} />
                Linked
              </span>
            ) : (
              <button
                className="secondary compact"
                disabled={!data.configured || busy || voiceActive}
                onClick={() => void act(() => startGoogle("link"))}
              >
                Link Google account
              </button>
            )}
          </div>
          <div className="setting-row">
            <span>
              <strong>Calendar access</strong>
              <small>
                Events and availability from your selected calendars.
              </small>
            </span>
            <button
              className="secondary compact"
              disabled={!data.configured || busy || voiceActive}
              onClick={() => void act(() => startGoogle("calendar"))}
            >
              {data.calendar_enabled
                ? "Reconnect Calendar"
                : "Connect Calendar"}
            </button>
          </div>
          {data.calendar_enabled && (
            <div className="setting-row">
              <span>
                <strong>Calendar editing</strong>
                <small>
                  {data.calendar_write_enabled
                    ? "Create, edit and delete personal events on writable calendars."
                    : "Allow Eri and the website to save changes to Google Calendar."}
                </small>
              </span>
              <button
                className="secondary compact"
                disabled={!data.configured || busy || voiceActive}
                onClick={() => void act(() => startGoogle("calendar_write"))}
              >
                {data.calendar_write_enabled
                  ? "Reconnect editing"
                  : "Enable Calendar editing"}
              </button>
            </div>
          )}
          {voiceActive && (
            <p className="footnote">End voice before opening Google sign-in.</p>
          )}
          {data.calendar_enabled && (
            <>
              <div className="calendar-sync-row">
                <span role="status">
                  {data.syncing
                    ? "Syncing calendars…"
                    : data.status === "needs_reconnect"
                      ? "Calendar permission expired. Reconnect to continue."
                      : data.status === "error"
                        ? "Sync could not finish. Cached events may be out of date."
                        : data.last_sync_at
                          ? "Last synced " +
                            new Date(data.last_sync_at).toLocaleString()
                          : "Waiting for the first sync…"}
                </span>
                <button
                  className="text-button"
                  disabled={
                    busy || data.syncing || data.status === "needs_reconnect"
                  }
                  onClick={() =>
                    void act(async () => {
                      await post("/integrations/google/sync");
                      setNotice("Calendar sync requested.");
                    })
                  }
                >
                  <RefreshCw size={14} />
                  Sync now
                </button>
              </div>
              <div className="google-calendar-choices">
                {data.calendars.map((cal) => (
                  <label key={cal.id} className="setting-row">
                    <span>
                      <strong>
                        {cal.title}
                        {cal.primary ? " · Primary" : ""}
                      </strong>
                      <small>
                        {cal.available
                          ? cal.timezone +
                            (cal.writable ? " · Can edit" : " · Read-only")
                          : "No longer accessible"}
                        {cal.selected && !cal.last_sync_at
                          ? " · Waiting for sync"
                          : ""}
                      </small>
                    </span>
                    <input
                      type="checkbox"
                      aria-label={"Use calendar " + cal.title}
                      checked={cal.selected}
                      disabled={busy || (!cal.available && !cal.selected)}
                      onChange={(e) => {
                        const selected = e.target.checked;
                        setData((current) =>
                          current
                            ? {
                                ...current,
                                calendars: current.calendars.map((row) =>
                                  row.id === cal.id
                                    ? { ...row, selected }
                                    : row,
                                ),
                              }
                            : current,
                        );
                        void act(async () => {
                          const result = await mutate(
                            "calendar.select",
                            {
                              calendar_id: cal.id,
                              expected_revision: cal.revision,
                              selected,
                            },
                            "Calendar selection saved",
                          );
                          if (!result)
                            throw new Error(
                              "Calendar selection could not be saved. Review the current selection and try again.",
                            );
                        });
                      }}
                    />
                  </label>
                ))}
              </div>
              {!disconnecting ? (
                <button
                  className="text-button danger"
                  disabled={busy}
                  onClick={() => setDisconnecting(true)}
                >
                  Disconnect Calendar
                </button>
              ) : (
                <div className="disconnect-confirm">
                  <p>
                    Remove synced events from Eri and revoke Calendar access?
                    Your Google events stay in Google.
                  </p>
                  <button
                    className="secondary compact"
                    disabled={busy}
                    onClick={() =>
                      void act(async () => {
                        const result = await post<{ revoked: boolean }>(
                          "/integrations/google/disconnect-calendar",
                        );
                        setDisconnecting(false);
                        setNotice(
                          result.revoked
                            ? "Calendar disconnected. Google sign-in stays linked."
                            : "Calendar disconnected locally. Remove Eridani’s access in your Google account to finish revoking the grant.",
                        );
                      })
                    }
                  >
                    Disconnect now
                  </button>
                  <button
                    className="text-button"
                    onClick={() => setDisconnecting(false)}
                  >
                    Keep connected
                  </button>
                </div>
              )}
            </>
          )}
          {data.linked &&
            (!unlinking ? (
              <button
                className="text-button danger"
                disabled={busy}
                onClick={() => setUnlinking(true)}
              >
                Unlink Google sign-in
              </button>
            ) : (
              <div className="disconnect-confirm">
                <p>
                  Unlink this account and disconnect Calendar? Devices signed in
                  through Google will need to pair again.
                </p>
                <button
                  className="secondary compact"
                  disabled={busy}
                  onClick={() =>
                    void act(async () => {
                      await post("/integrations/google/unlink");
                      location.assign("/");
                    })
                  }
                >
                  Unlink account
                </button>
                <button
                  className="text-button"
                  onClick={() => setUnlinking(false)}
                >
                  Keep account
                </button>
              </div>
            ))}
          <a
            className="footnote external-link"
            href="https://myaccount.google.com/connections"
            target="_blank"
            rel="noopener noreferrer"
          >
            Google account connections
            <ExternalLink size={12} />
          </a>
        </>
      )}
    </section>
  );
}
