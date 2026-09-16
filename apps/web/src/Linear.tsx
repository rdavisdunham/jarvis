import { useEffect, useState } from "react";
import { api, post } from "./api";
import type { Task } from "./types";

type Mutate = (
  tool: string,
  args: unknown,
  message: string,
) => Promise<unknown>;
export type LinearStatus = {
  connected: boolean;
  workspace?: string;
  revision: number;
  viewer_id?: string;
  team_ids: string[];
  only_mine: boolean;
  status: string;
  error: string;
  last_sync_at: string | null;
  teams: { id: string; name: string }[];
  states: { id: string; name: string; team: { id: string } }[];
  users: { id: string; name: string; active: boolean }[];
  projects: { id: string; name: string }[];
  recent_changes: { job_id: string; status: string; message: string }[];
};
export function LinearSettings({
  revision,
  mutate,
}: {
  revision: number;
  mutate: Mutate;
}) {
  const [data, setData] = useState<LinearStatus | null>(null);
  const [key, setKey] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const [refresh, setRefresh] = useState(0),
    [teams, setTeams] = useState<string[]>([]),
    [mine, setMine] = useState(true);
  const [dirty, setDirty] = useState(false),
    [disconnecting, setDisconnecting] = useState(false);
  useEffect(() => {
    let live = true;
    api<LinearStatus>("/integrations/linear")
      .then((d) => {
        if (!live) return;
        setData(d);
        if (!dirty) {
          setTeams(d.team_ids ?? []);
          setMine(d.only_mine ?? true);
        }
      })
      .catch((e) => {
        if (live) setError(e.message);
      });
    return () => {
      live = false;
    };
  }, [revision, refresh, dirty]);
  async function act(fn: () => Promise<unknown>) {
    setBusy(true);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
      setRefresh((n) => n + 1);
    }
  }
  return (
    <section className="google-settings">
      <h2>Linear</h2>
      <p>
        Bring issues into your task workspace. Changes to linked task titles,
        notes, status, due dates and assignees sync back to Linear.
      </p>
      {data?.connected && <div className="integration-summary"><strong>{data.workspace ?? "Linear connected"}</strong><span>{data.team_ids.length} teams · {data.recent_changes.filter(c => !["succeeded", "completed", "cancelled"].includes(c.status)).length} changes pending or needing attention</span><small>Last successful sync: {data.last_sync_at ? new Date(data.last_sync_at).toLocaleString() : "Not yet synced"}</small></div>}
      {error && (
        <p role="alert" className="error-banner">
          {error}
        </p>
      )}
      {!data ? (
        <p>Loading connection…</p>
      ) : !data.connected ? (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void act(async () => {
              await post("/integrations/linear/connect", { api_key: key });
              setKey("");
              setDirty(false);
            });
          }}
        >
          <label>
            Personal API key
            <input
              aria-label="Linear API key"
              type="password"
              autoComplete="off"
              required
              value={key}
              onChange={(e) => setKey(e.target.value)}
            />
          </label>
          <p className="footnote">
            Create a personal API key in Linear → Settings → Security & access,
            with read, create and write access for your chosen teams. Stored
            encrypted in Eridani.
          </p>
          <button className="secondary" disabled={busy || !key}>
            Connect Linear
          </button>
        </form>
      ) : (
        <>
          <p>
            <strong>{data.workspace}</strong> · {data.status}
            {data.last_sync_at
              ? " · Synced " + new Date(data.last_sync_at).toLocaleString()
              : ""}
          </p>
          {data.error && <p role="alert">{data.error}</p>}
          <fieldset disabled={busy}>
            <legend>Teams to sync</legend>
            {data.teams.map((t) => (
              <label className="event-checkbox" key={t.id}>
                <input
                  type="checkbox"
                  checked={teams.includes(t.id)}
                  onChange={(e) => {
                    setDirty(true);
                    setTeams((ids) =>
                      e.target.checked
                        ? [...ids, t.id]
                        : ids.filter((id) => id !== t.id),
                    );
                  }}
                />
                {t.name}
              </label>
            ))}
            <label className="event-checkbox">
              <input
                type="checkbox"
                checked={mine}
                onChange={(e) => {
                  setDirty(true);
                  setMine(e.target.checked);
                }}
              />
              Only issues assigned to me
            </label>
            <div className="dialog-actions">
              <button
                type="button"
                className="secondary"
                onClick={() =>
                  void act(async () => {
                    const saved = await mutate(
                      "linear.select",
                      {
                        expected_revision: data.revision,
                        team_ids: teams,
                        only_mine: mine,
                      },
                      "Linear sync queued",
                    );
                    if (saved) setDirty(false);
                    else
                      throw new Error(
                        "Selection was not saved. Refresh and try again.",
                      );
                  })
                }
              >
                Save sync scope
              </button>
              <button
                type="button"
                onClick={() =>
                  void act(() => post("/integrations/linear/sync", {}))
                }
              >
                Sync now
              </button>
            </div>
          </fieldset>
          <p className="footnote">
            Syncs every five minutes, with a full reconciliation daily. Alerts,
            precise due times and work blocks remain in Eridani. Removing an
            issue remotely keeps its local record for review.
          </p>
          <details>
            <summary>Recent Linear changes</summary>
            {data.recent_changes.length ? (
              data.recent_changes.map((j) => (
                <p key={j.job_id}>
                  {j.status} · {j.message}
                </p>
              ))
            ) : (
              <p>No changes yet.</p>
            )}
          </details>
          {!disconnecting ? (
            <button
              type="button"
              className="text-button"
              onClick={() => setDisconnecting(true)}
            >
              Disconnect Linear
            </button>
          ) : (
            <p>
              Keep local tasks and remove the stored key?
              <button
                type="button"
                disabled={busy}
                onClick={() =>
                  void act(async () => {
                    await post("/integrations/linear/disconnect", {});
                    setDisconnecting(false);
                  })
                }
              >
                Disconnect
              </button>
              <button type="button" onClick={() => setDisconnecting(false)}>
                Keep connected
              </button>
            </p>
          )}
        </>
      )}
    </section>
  );
}
type Comparison = {
  pending_change?: Record<string, string | number | null>;
  local: Task;
  linear: {
    title: string;
    description?: string;
    updatedAt: string;
    state: { name: string };
    dueDate?: string;
    assignee?: { name: string };
    project?: { name: string };
  } | null;
  edit_token: string;
};
export function LinearTask({
  task,
  mutate,
  onChanged,
}: {
  task: Task;
  mutate: Mutate;
  onChanged: () => void;
}) {
  const [data, setData] = useState<LinearStatus | null>(null),
    [team, setTeam] = useState("");
  const [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [compare, setCompare] = useState<Comparison | null>(null);
  const [state, setState] = useState(task.external?.state_id ?? ""),
    [priority, setPriority] = useState(task.external?.priority ?? 0);
  const [assignee, setAssignee] = useState<string | undefined>(undefined),
    [project, setProject] = useState<string | undefined>(undefined);
  useEffect(() => {
    let live = true;
    api<LinearStatus>("/integrations/linear")
      .then((d) => {
        if (live) {
          setData(d);
          setTeam(d.team_ids?.[0] ?? "");
        }
      })
      .catch(() => {});
    return () => {
      live = false;
    };
  }, []);
  async function act(fn: () => Promise<unknown>) {
    setBusy(true);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function change(tool: string, args: Record<string, unknown>) {
    if (
      await mutate(
        tool,
        { task_id: task.id, expected_revision: task.revision, ...args },
        "Linear change accepted",
      )
    )
      onChanged();
    else
      setError(
        "The result was not confirmed. Retry the same change to check its saved receipt.",
      );
  }
  if (
    task.id === "new" ||
    task.is_template ||
    (!data?.connected && !task.external?.provider)
  )
    return null;
  const linked = task.external?.provider === "linear",
    scopeTeam = task.external?.team_id ?? team;
  const needsReview =
    linked && !["synced", "pending"].includes(task.external?.sync_state ?? "");
  return (
    <section className="task-reminders linear-task">
      <h3>Linear</h3>
      {error && (
        <p role="alert" className="error-banner">
          {error}
        </p>
      )}
      {linked ? (
        <>
          <p>
            {task.external?.url?.startsWith("https://linear.app/") ? (
              <a href={task.external.url} target="_blank" rel="noreferrer">
                {task.external.identifier ?? "Open issue"}
              </a>
            ) : (
              task.external?.identifier
            )}{" "}
            · {task.external?.sync_state}
            {task.external?.state ? " · " + task.external.state : ""}
          </p>
          {!data?.connected && (
            <>
              <p>Reconnect Linear in Settings to edit shared fields.</p>
              <button
                type="button"
                onClick={() =>
                  void act(() => change("linear.resolve", { choice: "unlink" }))
                }
              >
                Unlink · keep local task
              </button>
            </>
          )}
          {data?.connected && (
            <>
              {task.external?.sync_state === "pending" && (
                <p role="status">
                  Waiting for Linear to confirm. Reopen this task to see the
                  result.
                </p>
              )}
              {!needsReview && task.external?.sync_state !== "pending" && (
                <details>
                  <summary>Linear workflow</summary>
                  <fieldset disabled={busy}>
                    <label>
                      Status
                      <select
                        aria-label="Linear status"
                        value={state}
                        onChange={(e) => setState(e.target.value)}
                      >
                        {data.states
                          ?.filter((s) => s.team.id === scopeTeam)
                          .map((s) => (
                            <option value={s.id} key={s.id}>
                              {s.name}
                            </option>
                          ))}
                      </select>
                    </label>
                    <label>
                      Priority
                      <select
                        aria-label="Linear priority"
                        value={priority}
                        onChange={(e) => setPriority(Number(e.target.value))}
                      >
                        {["None", "Urgent", "High", "Medium", "Low"].map(
                          (p, i) => (
                            <option key={i} value={i}>
                              {p}
                            </option>
                          ),
                        )}
                      </select>
                    </label>
                    <label>
                      Assignee
                      <select
                        value={assignee ?? "__keep"}
                        onChange={(e) => setAssignee(e.target.value)}
                      >
                        <option value="__keep">Keep current</option>
                        <option value="">Unassigned</option>
                        {data.users
                          ?.filter((u) => u.active)
                          .map((u) => (
                            <option key={u.id} value={u.id}>
                              {u.name}
                            </option>
                          ))}
                      </select>
                    </label>
                    <label>
                      Project
                      <select
                        value={project ?? "__keep"}
                        onChange={(e) => setProject(e.target.value)}
                      >
                        <option value="__keep">Keep current</option>
                        <option value="">No project</option>
                        {data.projects?.map((p) => (
                          <option key={p.id} value={p.id}>
                            {p.name}
                          </option>
                        ))}
                      </select>
                    </label>
                    <button
                      type="button"
                      onClick={() =>
                        void act(() =>
                          change("linear.update", {
                            state_id: state,
                            priority,
                            ...(assignee !== undefined && assignee !== "__keep"
                              ? { assignee_id: assignee || null }
                              : {}),
                            ...(project !== undefined && project !== "__keep"
                              ? { project_id: project || null }
                              : {}),
                          }),
                        )
                      }
                    >
                      Save Linear workflow
                    </button>
                  </fieldset>
                </details>
              )}
              {task.external?.sync_state !== "pending" && (
                <button
                  type="button"
                  onClick={() =>
                    void act(async () =>
                      setCompare(
                        await api<Comparison>(
                          "/linear/tasks/" + task.id + "/comparison",
                        ),
                      ),
                    )
                  }
                >
                  Review Linear copy
                </button>
              )}
              {compare && (
                <div className="sync-comparison">
                  <h4>Current Linear copy</h4>
                  {!!Object.keys(compare.pending_change ?? {}).length && (
                    <details>
                      <summary>Your unconfirmed change</summary>
                      {Object.entries(compare.pending_change ?? {}).map(
                        ([key, value]) => (
                          <p className="plain-details" key={key}>
                            {key}: {value ?? "None"}
                          </p>
                        ),
                      )}
                    </details>
                  )}
                  {compare.linear ? (
                    <>
                      <strong>{compare.linear.title}</strong>
                      <p className="plain-details">
                        {compare.linear.description || "No notes"}
                      </p>
                      <p>
                        {compare.linear.state.name} · Due{" "}
                        {compare.linear.dueDate || "not set"} ·{" "}
                        {compare.linear.assignee?.name || "Unassigned"} ·{" "}
                        {compare.linear.project?.name || "No project"}
                      </p>
                    </>
                  ) : (
                    <p>
                      This issue is no longer available. Your local task is
                      preserved.
                    </p>
                  )}
                  <p>
                    Local task: {task.title} · {task.status} · Due{" "}
                    {task.due_date || "not set"}
                  </p>
                  {compare.linear && (
                    <>
                      <button
                        type="button"
                        disabled={busy}
                        onClick={() =>
                          void act(() =>
                            change("linear.resolve", {
                              choice: "linear",
                              edit_token: compare.edit_token,
                            }),
                          )
                        }
                      >
                        Use Linear version
                      </button>
                      <button
                        type="button"
                        disabled={busy}
                        onClick={() =>
                          void act(() =>
                            change("linear.resolve", {
                              choice: "eridani",
                              edit_token: compare.edit_token,
                            }),
                          )
                        }
                      >
                        Keep Eridani version
                      </button>
                    </>
                  )}
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() =>
                      void act(() =>
                        change("linear.resolve", { choice: "unlink" }),
                      )
                    }
                  >
                    Unlink · keep local task
                  </button>
                </div>
              )}
            </>
          )}
        </>
      ) : (
        <>
          <p>
            Publish this task as a new Linear issue. It will be assigned to you.
          </p>
          <label>
            Team
            <select
              aria-label="Linear team"
              value={team}
              onChange={(e) => setTeam(e.target.value)}
            >
              {data?.teams
                .filter((t) => data.team_ids.includes(t.id))
                .map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.name}
                  </option>
                ))}
            </select>
          </label>
          <button
            type="button"
            disabled={busy || !team}
            onClick={() =>
              void act(() => change("linear.publish", { team_id: team }))
            }
          >
            Publish to Linear
          </button>
        </>
      )}
    </section>
  );
}
