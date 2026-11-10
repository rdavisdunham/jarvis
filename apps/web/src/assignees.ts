import { useEffect, useState } from "react";
import { api } from "./api";
import type { Actor } from "./productivity";
import { useTaskCardHost } from "./task-card-host";

/** Who a task can be assigned to: the signed-in account (or the shared workspace's members), the assistant, or no one. */
export type AssigneeOption = { key: string; actorId: string | null; name: string; label: string; group: "People" | "Assistant" | "" };
export type WorkspaceMember = { account_id: string; name: string; role: string; active: boolean; actor_id: string | null };
/** The account the app is signed in as; `workspaceId` is null in the personal workspace. */
export type AccountContext = { name: string; accountId?: string; workspaceId: string | null };

export const UNASSIGNED = "unassigned";
const ASSISTANT = "Dani";

export function assigneeOptions({actors, account, members}: {actors: Actor[]; account?: AccountContext; members: WorkspaceMember[] | null}): AssigneeOption[] {
  const live = actors.filter(a => !a.archived);
  const byId = new Map(actors.map(a => [a.id, a]));
  const people: AssigneeOption[] = members
    ? members.filter(m => m.active && m.actor_id).map(m => ({
        key: m.actor_id!, actorId: m.actor_id, name: byId.get(m.actor_id!)?.name ?? m.name,
        label: m.name + (m.account_id === account?.accountId ? " (me)" : ""), group: "People" as const,
      }))
    : [(owner => ({ key: owner?.id ?? "owner", actorId: owner?.id ?? null, name: "owner", label: account?.name ? account.name + " (me)" : "Me", group: "People" as const }))(live.find(a => a.name === "owner"))];
  const agents = live.filter(a => a.kind === "agent").map(a => ({ key: a.id, actorId: a.id, name: a.name, label: a.name, group: "Assistant" as const }));
  // A workspace that never assigned work to the assistant has no agent actor yet; the name creates it.
  const assistant = agents.length ? agents : [{ key: ASSISTANT, actorId: null, name: ASSISTANT, label: ASSISTANT, group: "Assistant" as const }];
  const none = live.find(a => a.name.toLowerCase() === UNASSIGNED);
  return [...people, ...assistant, { key: none?.id ?? UNASSIGNED, actorId: none?.id ?? null, name: none?.name ?? UNASSIGNED, label: "Unassigned", group: "" }];
}

/** The option for a task's current assignee, or a label for one who isn't offered (a Linear person on a synced task). */
export function currentAssignee(options: AssigneeOption[], value: {id?: string | null; name?: string | null}, actors: Actor[]) {
  const match = options.find(o => (value.id && o.actorId === value.id) || (!value.id && o.name === (value.name || "owner")));
  if (match) return { option: match, label: match.label };
  const name = (value.id && actors.find(a => a.id === value.id)?.name) || value.name || "";
  return { option: null, label: name.toLowerCase() === UNASSIGNED ? "Unassigned" : name || "Unassigned" };
}

export function filterOptions(options: AssigneeOption[], query: string) {
  const q = query.trim().toLowerCase();
  return q ? options.filter(o => o.label.toLowerCase().includes(q) || o.name.toLowerCase().includes(q)) : options;
}

const memberCache = new Map<string, Promise<WorkspaceMember[]>>();
/** Options for the signed-in account's workspace. Members load once per shared workspace. */
export function useAssigneeOptions(actors?: Actor[]) {
  const host = useTaskCardHost();
  const account = host?.account;
  const all = actors ?? host?.actors ?? [];
  const [members, setMembers] = useState<WorkspaceMember[] | null>(null);
  useEffect(() => {
    const ws = account?.workspaceId;
    if (!ws) { setMembers(null); return; }
    let live = true;
    if (!memberCache.has(ws)) memberCache.set(ws, api<{members: WorkspaceMember[]}>("/accounts/members/" + ws).then(d => d.members).catch(e => { memberCache.delete(ws); throw e; }));
    void memberCache.get(ws)!.then(m => { if (live) setMembers(m); }).catch(() => { if (live) setMembers(null); });
    return () => { live = false; };
  }, [account?.workspaceId]);
  return { options: assigneeOptions({ actors: all, account, members }), actors: all };
}
