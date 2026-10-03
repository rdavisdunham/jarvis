import { useEffect, useRef, useState } from "react";
import { api, post } from "./api";
import { startGoogle } from "./GoogleSettings";
import { SettingRow, SettingsGroup } from "./SettingsLayout";
// Imported here (after App.tsx pulls in shell.css) so settings rules win over the shell defaults.
import "./settings.css";
type Workspace = { id: string; name: string; kind: string; role: string };
type Invite = {
  id: string;
  workspace_id: string | null;
  workspace: string;
  email: string;
  role: string;
  status: string;
  expires_at: string;
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
              {w.name} ({w.role})
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
  const inviteId = new URLSearchParams(location.search).get("invite");
  const [focusedInvite, setFocusedInvite] = useState<Invite | null>(null), [inviteError, setInviteError] = useState("");
  const [invitationMessage, setInvitationMessage] = useState("");
  async function loadInvitation() {
    if (!inviteId) return;
    try { setFocusedInvite(await api<Invite>("/accounts/invitations/"+encodeURIComponent(inviteId))); setInviteError(""); }
    catch (e) { setInviteError((e as Error).message); }
  }
  useEffect(() => { void loadInvitation(); }, [inviteId]);
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
      await loadInvitation();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const owned = data?.workspaces.filter((w) => w.role === "owner") ?? [];
  return (
    <div className="sharing-settings">
      <SettingsGroup title="People & sharing" description="Assignment organizes responsibility; it never grants access.">
        <div className="sharing-overview"><div><strong>Personal workspace</strong><p>Your own tasks, notes, memory and connected accounts.</p></div><div><strong>Shared workspace</strong><p>A separate set of tasks, notes and local calendar entries for invited people.</p></div></div>
        {error && (
          <p className="error-banner" role="alert">
            {error}
          </p>
        )}
        {message && <p role="status" className="settings-callout">{message}</p>}
        <p className="footnote">
          Shared workspaces start empty. Switch to one before adding shared work.
          Private records and external calendars are never moved automatically.
          Shared chats are temporary and don’t teach personal memory. Google and
          Linear connections are managed in Personal.
        </p>
      </SettingsGroup>
      {inviteId && <SettingsGroup className="invitation-card" title="Your invitation" description={"Signed in as " + (data?.email ?? data?.name ?? "") + "."}>
        {inviteError ? <><p role="alert" className="error-banner">{inviteError}</p><div className="setting-actions"><button className="btn" onClick={() => void startGoogle("login").catch(e => setInviteError(e.message))}>Use another Google account</button></div></> : focusedInvite ? <>
          <div className="settings-item"><div className="settings-item-main"><strong className="settings-item-title">{focusedInvite.workspace}</strong><p>{focusedInvite.workspace_id?`Join only ${focusedInvite.workspace} as ${focusedInvite.role}. Your Personal records stay separate.`:"This gives you your own Personal account with your own records, not access to the inviter’s records."}</p><span className="settings-item-meta"><span>{focusedInvite.email}</span><span className="chip">{humanize(focusedInvite.role)}</span></span></div>
          {focusedInvite.status === "pending" && <div className="settings-item-actions"><button className="btn btn-primary" disabled={busy} onClick={() => void act(async () => {
            await post("/accounts/accept", {invite_id:focusedInvite.id}); setMessage(focusedInvite.workspace_id?"Invitation accepted. Choose this shared workspace in the navigation menu.":"Your Personal account is ready. Its records belong to you.");
          })}>Accept invitation</button></div>}</div>
          {focusedInvite.status !== "pending" && <p className="settings-callout">{focusedInvite.status === "accepted" ? (focusedInvite.workspace_id?"Already accepted. Choose this shared workspace in the navigation menu.":"Your own Personal account is ready; your records are separate from the inviter’s.") : "This invitation expired or was revoked. Ask its sender for a new one."}</p>}
        </> : <p role="status" className="settings-empty">Checking your invitation…</p>}
      </SettingsGroup>}
      {!!data?.invitations.filter(i => i.id !== inviteId).length && (
        <SettingsGroup title="Invitations for you">
          <div className="settings-list">
          {data.invitations.filter(i => i.id !== inviteId).map((i) => (
            <div className="settings-item" key={i.id}>
              <span className="settings-item-main">
                <strong className="settings-item-title">{i.workspace}</strong><span className="setting-hint">{i.workspace_id?`Join this workspace as ${i.role}; Personal records stay separate.`:"Your own account and records; no access to the inviter’s data."}</span>
                <span className="settings-item-meta"><span>{i.email}</span><span className="chip">{humanize(i.role)}</span></span>
              </span>
              <span className="settings-item-actions"><button
                className="btn btn-primary btn-sm"
                disabled={busy}
                onClick={() =>
                  void act(async () => {
                    await post("/accounts/accept", { invite_id: i.id });
                    setMessage(
                      i.workspace_id?"Invitation accepted. Choose this shared workspace in the navigation menu.":"Your Personal account is ready with its own records.",
                    );
                  })
                }
              >
                Accept invitation
              </button></span>
            </div>
          ))}
          </div>
        </SettingsGroup>
      )}
      <SettingsGroup title="Create a shared workspace" description="A new, empty workspace you can invite people to.">
      <form
        className="settings-form"
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
          <label className="field">
            <span className="field-label-text">Name</span>
            <input
              aria-label="Shared workspace name"
              required
              maxLength={200}
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </label>
          <label className="field">
            <span className="field-label-text">Kind</span>
            <select
              aria-label="Shared workspace kind"
              value={kind}
              onChange={(e) => setKind(e.target.value)}
            >
              <option value="space">Space</option>
              <option value="project">Project</option>
            </select>
          </label>
          <div className="settings-form-actions"><button className="btn btn-primary" disabled={busy || !name.trim()}>
            Create workspace
          </button></div>
      </form>
      </SettingsGroup>
      <SettingsGroup title="Invite people and manage access" description="No email is sent. After creating an invitation, copy its message and share it with the person yourself.">
        <SettingRow label="Manage sharing for" hint="Choose the workspace whose people you want to manage.">
          <select
            aria-label="Manage sharing"
            value={target}
            onChange={(e) => setTarget(e.target.value)}
          >
            <option value="">
              {data?.can_invite_accounts
                ? "Own separate personal account"
                : "Choose a workspace"}
            </option>
            {owned.map((w) => (
              <option value={w.id} key={w.id}>
                {w.name}
              </option>
            ))}
          </select>
        </SettingRow>
        <form
          className="settings-form"
          onSubmit={(e) => {
            e.preventDefault();
            void act(async () => {
              const created = await post<Invite>("/accounts/invitations", {
                workspace_id: target || null,
                email,
                role,
              });
              setInvitationMessage("You’re invited to " + created.workspace + " in Eridani. Sign in with " + created.email + ". " + location.origin + "/?view=settings&sharing=1&invite=" + created.id);
              setEmail("");
              setMessage(
                "Invitation ready. Share this app’s address with them; they must sign in with that Google email. No email was sent.",
              );
            });
          }}
        >
          <label className="field">
            <span className="field-label-text">Google email</span>
            <input
              aria-label="Invite Google email"
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </label>
          <label className="field">
            <span className="field-label-text">Role</span>
            <select
              aria-label="Invitation role"
              value={role}
              onChange={(e) => setRole(e.target.value)}
            >
              <option value="editor">Editor</option>
              <option value="viewer">Viewer</option>
            </select>
          </label>
          <div className="settings-form-actions">
            <button className="btn btn-primary" disabled={busy || (!target && !data?.can_invite_accounts)}>
              Create invitation
            </button>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() =>
                void navigator.clipboard
                  .writeText(location.origin + "/?view=settings&sharing=1")
                  .then(() => setMessage("Sign-in link copied."))
                  .catch(() => setMessage("Share this address: " + location.origin))
              }
            >
              Copy sign-in link
            </button>
          </div>
        </form>
        {invitationMessage && <div className="invitation-copy"><label className="field"><span className="field-label-text">Invitation message</span><textarea readOnly value={invitationMessage}/></label>
          <button className="btn btn-soft btn-sm" style={{alignSelf:"flex-start"}} onClick={() => void navigator.clipboard.writeText(invitationMessage)
            .then(() => setMessage("Invitation message copied.")).catch(() => setMessage("Select and copy the invitation message above."))}>Copy invitation</button></div>}
        <p className="role-guide"><strong>Viewer:</strong> can read. <strong>Editor:</strong> can add and change records. <strong>Owner:</strong> also manages invitations and access.</p>
        {!!members.length && <h4>Members</h4>}
        <div className="settings-list">
        {members.map((m) => (
          <div className="settings-item" key={m.account_id}>
            <span className="settings-item-main">
              <strong className="settings-item-title">{m.name}</strong>
              <span className="settings-item-meta"><span className="chip">{m.active ? humanize(m.role) : "Access revoked"}</span></span>
            </span>
            {m.role !== "owner" && (
              <span className="settings-item-actions">
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
                    className="btn btn-danger btn-sm"
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
              </span>
            )}
          </div>
        ))}
        </div>
        {!!invites.length && <h4>Pending invitations</h4>}
        <div className="settings-list">
        {invites.map((i) => (
          <div className="settings-item" key={i.id}>
            <span className="settings-item-main">
              <span className="settings-item-title">{i.email}</span>
              <span className="settings-item-meta"><span className="chip">Pending</span><span className="chip">{humanize(i.role)}</span><span>Expires {new Date(i.expires_at).toLocaleDateString()}</span></span>
            </span>
            <span className="settings-item-actions"><button
              className="btn btn-ghost btn-sm"
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
            </button></span>
          </div>
        ))}
        </div>
      </SettingsGroup>
    </div>
  );
}
const humanize = (value: string) => value.charAt(0).toUpperCase() + value.slice(1).replaceAll("_", " ");
