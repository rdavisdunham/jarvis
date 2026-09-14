import { useEffect, useRef, useState } from "react";
import { api, post } from "./api";
type Workspace = { id: string; name: string; kind: string; role: string };
type Invite = {
  id: string;
  workspace_id: string | null;
  workspace: string;
  email: string;
  role: string;
  status: string;
};
type Account = {
  account_id: string;
  name: string;
  email: string | null;
  active_workspace_id: string | null;
  workspaces: Workspace[];
  invitations: Invite[];
  outgoing: Invite[];
  can_invite_accounts: boolean;
};
type Member = {
  account_id: string;
  name: string;
  role: string;
  active: boolean;
  revision: number;
};
const changed = () => window.dispatchEvent(new Event("eri-accounts-changed"));
export function AccountSwitcher({
  onSwitch,
  onSharing,
}: {
  onSwitch: (id: string | null) => Promise<void>;
  onSharing: () => void;
}) {
  const [data, setData] = useState<Account | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    const load = () => {
      void api<Account>("/accounts")
        .then(setData)
        .catch((e) => setError(e.message));
    };
    load();
    window.addEventListener("eri-accounts-changed", load);
    return () => window.removeEventListener("eri-accounts-changed", load);
  }, []);
  if (!data) return null;
  return (
    <div className="account-switcher">
      <label>
        <span className="sr-only">Active workspace</span>
        <select
          aria-label="Active workspace"
          value={data.active_workspace_id ?? ""}
          onChange={(e) =>
            void onSwitch(e.target.value || null).catch((e) =>
              setError(e.message),
            )
          }
        >
          <option value="">Personal</option>
          {data.workspaces.map((w) => (
            <option key={w.id} value={w.id}>
              {w.name} · {w.role}
            </option>
          ))}
        </select>
      </label>
      {data.invitations.length > 0 && (
        <button className="text-button" onClick={onSharing}>
          {data.invitations.length} invitation
          {data.invitations.length === 1 ? "" : "s"}
        </button>
      )}
      {error && <p role="alert">{error}</p>}
    </div>
  );
}
export function SharingSettings() {
  const [data, setData] = useState<Account | null>(null),
    [target, setTarget] = useState(""),
    [members, setMembers] = useState<Member[]>([]),
    [invites, setInvites] = useState<Invite[]>([]);
  const [name, setName] = useState(""),
    [kind, setKind] = useState("space"),
    [email, setEmail] = useState(""),
    [role, setRole] = useState("editor");
  const [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [message, setMessage] = useState("");
  const creation = useRef<{ key: string; id: string } | null>(null);
  async function load() {
    const d = await api<Account>("/accounts");
    setData(d);
    changed();
  }
  useEffect(() => {
    void load().catch((e) => setError(e.message));
  }, []);
  async function refreshMembers() {
    if (!target) {
      setMembers([]);
      const d = await api<Account>("/accounts");
      setInvites(d.outgoing ?? []);
      return;
    }
    const d = await api<{ members: Member[]; invitations: Invite[] }>(
      "/accounts/members/" + target,
    );
    setMembers(d.members);
    setInvites(d.invitations);
  }
  useEffect(() => {
    void refreshMembers().catch((e) => setError(e.message));
  }, [target]);
  async function act(fn: () => Promise<void>) {
    setBusy(true);
    setError("");
    setMessage("");
    try {
      await fn();
      await load();
      await refreshMembers();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const owned = data?.workspaces.filter((w) => w.role === "owner") ?? [];
  return (
    <section className="sharing-settings">
      <h2>People & sharing</h2>
      <p className="footnote">
        Your personal tasks, notes, memory and connected accounts stay private.
        Create a shared space or project, then invite people to it. Everyone in
        that workspace sees its tasks, notes, local calendar and task alerts.
        Assignment alone never grants access.
      </p>
      {error && (
        <p className="error-banner" role="alert">
          {error}
        </p>
      )}
      {message && <p role="status">{message}</p>}
      {!!data?.invitations.length && (
        <section>
          <h3>Invitations for you</h3>
          {data.invitations.map((i) => (
            <div className="sharing-row" key={i.id}>
              <span>
                <strong>{i.workspace}</strong>
                <small>
                  {i.email} · {i.role}
                </small>
              </span>
              <button
                disabled={busy}
                onClick={() =>
                  void act(async () => {
                    await post("/accounts/accept", { invite_id: i.id });
                    setMessage(
                      "Invitation accepted. Choose the workspace in the navigation menu.",
                    );
                  })
                }
              >
                Accept invitation
              </button>
            </div>
          ))}
        </section>
      )}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          void act(async () => {
            const key = JSON.stringify({ name, kind });
            if (creation.current?.key !== key)
              creation.current = { key, id: crypto.randomUUID() };
            const w = await post<Workspace>("/accounts/workspaces", {
              name,
              kind,
              id: creation.current.id,
            });
            creation.current = null;
            setName("");
            setTarget(w.id);
            setMessage(
              "Shared workspace created. Your existing personal records remain private.",
            );
          });
        }}
      >
        <h3>Create a shared workspace</h3>
        <div className="sharing-fields">
          <label>
            Name
            <input
              aria-label="Shared workspace name"
              required
              maxLength={200}
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </label>
          <label>
            Kind
            <select
              aria-label="Shared workspace kind"
              value={kind}
              onChange={(e) => setKind(e.target.value)}
            >
              <option value="space">Space</option>
              <option value="project">Project</option>
            </select>
          </label>
          <button className="primary compact" disabled={busy || !name.trim()}>
            Create workspace
          </button>
        </div>
      </form>
      <section>
        <h3>Members & invitations</h3>
        <label>
          Manage sharing
          <select
            aria-label="Manage sharing"
            value={target}
            onChange={(e) => setTarget(e.target.value)}
          >
            <option value="">
              {data?.can_invite_accounts
                ? "Standalone personal account"
                : "Choose a workspace"}
            </option>
            {owned.map((w) => (
              <option value={w.id} key={w.id}>
                {w.name}
              </option>
            ))}
          </select>
        </label>
        <form
          className="sharing-fields"
          onSubmit={(e) => {
            e.preventDefault();
            void act(async () => {
              await post<Invite>("/accounts/invitations", {
                workspace_id: target || null,
                email,
                role,
              });
              setEmail("");
              setMessage(
                "Invitation ready. Share this app’s address with them; they must sign in with that Google email. No email was sent.",
              );
            });
          }}
        >
          <label>
            Google email
            <input
              aria-label="Invite Google email"
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </label>
          <label>
            Role
            <select
              aria-label="Invitation role"
              value={role}
              onChange={(e) => setRole(e.target.value)}
            >
              <option value="editor">Editor</option>
              <option value="viewer">Viewer</option>
            </select>
          </label>
          <button disabled={busy || (!target && !data?.can_invite_accounts)}>
            Create invitation
          </button>
        </form>
        <button
          className="text-button"
          onClick={() =>
            void navigator.clipboard
              .writeText(location.origin + "/?view=settings&sharing=1")
              .then(() => setMessage("Sign-in link copied."))
              .catch(() => setMessage("Share this address: " + location.origin))
          }
        >
          Copy sign-in link
        </button>
        {members.map((m) => (
          <div className="sharing-row" key={m.account_id}>
            <span>
              <strong>{m.name}</strong>
              <small>{m.active ? m.role : "Access revoked"}</small>
            </span>
            {m.role !== "owner" && (
              <>
                <select
                  aria-label={"Role for " + m.name}
                  disabled={busy || !m.active}
                  value={m.role}
                  onChange={(e) =>
                    void act(async () => {
                      await post("/accounts/members", {
                        workspace_id: target,
                        account_id: m.account_id,
                        expected_revision: m.revision,
                        active: m.active,
                        role: e.target.value,
                      });
                    })
                  }
                >
                  <option value="editor">Editor</option>
                  <option value="viewer">Viewer</option>
                </select>
                {m.active && (
                  <button
                    disabled={busy}
                    onClick={() =>
                      void act(async () => {
                        await post("/accounts/members", {
                          workspace_id: target,
                          account_id: m.account_id,
                          expected_revision: m.revision,
                          active: false,
                          role: m.role,
                        });
                      })
                    }
                  >
                    Revoke access
                  </button>
                )}
              </>
            )}
          </div>
        ))}
        {invites.map((i) => (
          <div className="sharing-row" key={i.id}>
            <span>
              {i.email}
              <small>Pending · {i.role} · expires in seven days</small>
            </span>
            <button
              disabled={busy}
              onClick={() =>
                void act(async () => {
                  await post("/accounts/revoke-invitation", {
                    invite_id: i.id,
                  });
                })
              }
            >
              Revoke invitation
            </button>
          </div>
        ))}
      </section>
      <p className="footnote">
        Shared workspaces start empty. Switch to one before adding shared work.
        Private records and external calendars are never moved automatically.
        Shared chats are temporary and don’t teach personal memory. Google and
        Linear connections are managed in Personal.
      </p>
    </section>
  );
}
