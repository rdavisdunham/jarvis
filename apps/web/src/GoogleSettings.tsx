import { useEffect, useState } from "react";
import { Check, ExternalLink, RefreshCw } from "lucide-react";
import { api, post } from "./api";
import { SettingRow, SettingsGroup } from "./SettingsLayout";

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
  const return_to = purpose === "login" ? location.pathname + location.search : "/?view=settings&section=integrations";
  const result = await post<{ url: string }>("/auth/google/start", { purpose, return_to });
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
  const selectedCount = data ? data.calendars.filter(c => c.selected && c.available).length : 0;
  return (
    <SettingsGroup className="google-settings" title="Google"
      description="Sign in with your account and bring your calendars into Eri’s view of your day.">
      {error && (
        <p className="error-banner" role="alert">
          <span>{error}</span>
          <button
            className="btn btn-sm"
            onClick={() => {
              setError("");
              setRefresh((n) => n + 1);
            }}
          >
            Retry
          </button>
        </p>
      )}
      {notice && <p role="status" className="settings-callout">{notice}</p>}
      {!data ? (
        <p role="status" className="settings-empty">Loading connection…</p>
      ) : (
        <>
          <div className="integration-status">
            <span className={"status-dot" + (data.linked ? " on" : "")} aria-hidden="true"/>
            <strong>{data.linked ? data.email : "Not connected"}</strong>
            {data.calendar_enabled
              ? <><span className="chip">{selectedCount} {selectedCount === 1 ? "calendar" : "calendars"}</span><span className="chip">{data.calendar_write_enabled ? "Read & edit" : "Read only"}</span></>
              : <span className="chip">Calendar access not enabled</span>}
          </div>
          {!data.configured && (
            <p className="settings-callout integration-hint">
              Google connection is not configured. Contact the app owner to enable it.
            </p>
          )}
          <SettingRow label="Sign in with Google" hint={data.linked ? data.email : "Connect your Google account to Eridani."}>
            {data.linked ? (
              <span className="connection-label">
                <Check size={16} />
                Linked
              </span>
            ) : (
              <button
                className="btn"
                disabled={!data.configured || busy || voiceActive}
                onClick={() => void act(() => startGoogle("link"))}
              >
                Link Google account
              </button>
            )}
          </SettingRow>
          <SettingRow label="Calendar access" hint="Events and availability from your selected calendars. Sign-in and Calendar permissions are separate.">
            <button
              className="btn"
              disabled={!data.configured || busy || voiceActive}
              onClick={() => void act(() => startGoogle("calendar"))}
            >
              {data.calendar_enabled
                ? "Reconnect Calendar"
                : "Connect Calendar"}
            </button>
          </SettingRow>
          {data.calendar_enabled && (
            <SettingRow label="Calendar editing" hint={data.calendar_write_enabled
              ? "Create, edit and delete personal events on calendars your Google account can write to."
              : "Allow Eri and the website to save changes to Google Calendar."}>
              <button
                className="btn"
                disabled={!data.configured || busy || voiceActive}
                onClick={() => void act(() => startGoogle("calendar_write"))}
              >
                {data.calendar_write_enabled
                  ? "Reconnect editing"
                  : "Enable Calendar editing"}
              </button>
            </SettingRow>
          )}
          {voiceActive && (
            <p className="footnote">End voice before opening Google sign-in.</p>
          )}
          {data.calendar_enabled && (
            <>
              <SettingRow label="Sync" hint={<span role="status">
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
                </span>}>
                <button
                  className="btn btn-ghost"
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
              </SettingRow>
              <div className="settings-block google-calendar-choices">
                <h4>Calendars</h4>
                {data.calendars.map((cal) => (
                  <label key={cal.id} className="setting-row switch-row">
                    <span className="setting-text">
                      <strong className="setting-label">
                        {cal.title}
                      </strong>
                      <span className="settings-item-meta">
                        {cal.primary && <span className="chip">Primary</span>}
                        {cal.available
                          ? <><span className="chip">{cal.writable ? "Can edit" : "Read only"}</span><span>{cal.timezone}</span></>
                          : <span className="chip chip-due-overdue">No longer accessible</span>}
                        {cal.selected && !cal.last_sync_at && <span>Waiting for sync</span>}
                      </span>
                    </span>
                    <input
                      type="checkbox"
                      className="switch"
                      role="switch"
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
              {disconnecting && (
                <div className="settings-confirm">
                  <p>
                    Remove synced events from Eri and revoke Calendar access?
                    Your Google events stay in Google.
                  </p>
                  <button
                    className="btn btn-danger"
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
                    className="btn btn-ghost"
                    onClick={() => setDisconnecting(false)}
                  >
                    Keep connected
                  </button>
                </div>
              )}
            </>
          )}
          {data.linked && unlinking && (
              <div className="settings-confirm">
                <p>
                  Unlink this account and disconnect Calendar? Devices signed in
                  through Google will need to pair again.
                </p>
                <button
                  className="btn btn-danger"
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
                  className="btn btn-ghost"
                  onClick={() => setUnlinking(false)}
                >
                  Keep account
                </button>
              </div>
            )}
          <div className="settings-links">
            {data.calendar_enabled && !disconnecting && (
              <button
                className="btn btn-danger btn-sm"
                disabled={busy}
                onClick={() => setDisconnecting(true)}
              >
                Disconnect Calendar
              </button>
            )}
            {data.linked && !unlinking && (
              <button
                className="btn btn-danger btn-sm"
                disabled={busy}
                onClick={() => setUnlinking(true)}
              >
                Unlink Google sign-in
              </button>
            )}
            <a
              className="external-link"
              href="https://myaccount.google.com/connections"
              target="_blank"
              rel="noopener noreferrer"
            >
              Google account connections
              <ExternalLink size={12} />
            </a>
          </div>
        </>
      )}
    </SettingsGroup>
  );
}
