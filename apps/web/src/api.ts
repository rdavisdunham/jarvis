import { recordRequest } from "./bug-report";
let csrf = "";
let device = "";
export function setDevice(value: string) {
  device = value;
}
export function setCsrf(value: string) {
  csrf = value;
}
export class ApiError extends Error {
  constructor(
    public code: string,
    message: string,
    public data?: unknown,
  ) {
    super(message);
  }
}
export async function api<T>(
  url: string,
  options: RequestInit = {},
): Promise<T> {
  let response: Response;
  const method = (options.method ?? "GET").toUpperCase(), started = performance.now();
  // The Report a bug ring buffer: method, path (query values dropped), status, timing; never bodies or headers.
  const note = (status: number, error?: string) => recordRequest({
    method, path: url, status, ms: Math.round(performance.now() - started), error,
    request_id: response?.headers?.get?.("x-request-id") ?? response?.headers?.get?.("x-railway-request-id") ?? undefined,
  });
  try {
    response = await fetch("/api/v1" + url, {
      credentials: "same-origin",
      ...options,
      headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": csrf,
        ...(device ? { "X-Device-Id": device } : {}),
        ...options.headers,
      },
    });
  } catch {
    note(0, "NETWORK");
    throw new ApiError(
      "NETWORK",
      "Connection lost. Your request may have saved. Retry the same request to check.",
    );
  }
  let value;
  try {
    value = await response.json();
  } catch {
    note(response.status, "BAD_RESPONSE");
    throw new ApiError(
      "NETWORK",
      "Eridani is reconnecting. Retry the same request to check whether it saved.",
    );
  }
  note(response.status, response.ok ? undefined : value?.error?.code ?? "ERROR");
  if (
    !response.ok &&
    ["ACCESS_REVOKED", "WORKSPACE_CHANGED"].includes(value.error?.code)
  ) {
    window.dispatchEvent(
      new CustomEvent("eri-access-ended", { detail: value.error.code }),
    );
  }
  if (!response.ok)
    throw new ApiError(
      value.error?.code ?? "ERROR",
      value.error?.message ??
        value.detail?.[0]?.msg ??
        "Something went wrong. Please try again.",
      value.error?.data,
    );
  return value as T;
}
export function post<T>(url: string, data: unknown = {}) {
  return api<T>(url, { method: "POST", body: JSON.stringify(data) });
}
const pause = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
export type ActionReceipt = { id: string; url: string; body: Record<string, unknown>; status: "queued" | "saved" | "delivered" | "uncertain" | "rejected"; error?: string; updatedAt: string; retention?: "tab" };
const ACTION_STORAGE = "eri-remotea-actions-v1";
let actionCache: ActionReceipt[] | null = null;
const actionListeners = new Set<() => void>();
export function actionReceipts(): ActionReceipt[] {
  if (!actionCache) {
    try { const value = JSON.parse(localStorage.getItem(ACTION_STORAGE) || "[]"); actionCache = Array.isArray(value) ? value.filter(x => x && typeof x.id === "string" && typeof x.url === "string" && x.body && typeof x.status === "string") : []; }
    catch { actionCache = []; }
  }
  return actionCache!;
}
export function subscribeActions(listener: () => void) { actionListeners.add(listener); return () => { actionListeners.delete(listener); }; }
function saveAction(receipt: ActionReceipt) {
  const all = [...actionReceipts().filter(x => x.id !== receipt.id), receipt];
  const settled = new Set(all.filter(x => ["delivered", "rejected"].includes(x.status)).slice(-100).map(x => x.id));
  actionCache = all.filter(x => !["delivered", "rejected"].includes(x.status) || settled.has(x.id));
  try { localStorage.setItem(ACTION_STORAGE, JSON.stringify(actionCache)); } catch { actionCache = actionCache.map(x => x.id === receipt.id ? { ...x, retention: "tab" } : x); }
  actionListeners.forEach(listener => listener());
}
function canonical(value: unknown): string {
  if (Array.isArray(value)) return "[" + value.map(canonical).join(",") + "]";
  if (value && typeof value === "object") return "{" + Object.entries(value).filter(([, v]) => v !== undefined).sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) => JSON.stringify(k) + ":" + canonical(v)).join(",") + "}";
  return JSON.stringify(value);
}
/** An unresolved intent keeps its ID across automatic retries, manual retries and reloads. */
export async function postAction<T>(url: string, data: Record<string, unknown> = {}, { retries = 2, delayMs = 500 }: { retries?: number; delayMs?: number } = {}): Promise<T> {
  const supplied = typeof data.action_id === "string" ? data.action_id : null;
  const intent = { ...data }; delete intent.action_id;
  const previous = actionReceipts().find(x => x.url === url && ["queued", "uncertain"].includes(x.status) && canonical(Object.fromEntries(Object.entries(x.body).filter(([k]) => k !== "action_id"))) === canonical(intent));
  const id = supplied ?? previous?.id ?? crypto.randomUUID();
  const body = { ...intent, action_id: id };
  const receipt: ActionReceipt = { id, url, body, status: previous?.status ?? "queued", updatedAt: new Date().toISOString() };
  saveAction(receipt);
  for (let attempt = 0; ; attempt++) {
    try {
      if (typeof navigator !== "undefined" && navigator.onLine === false) throw new ApiError("NETWORK", "Queued on this device until the connection returns.");
      const result = await post<T>(url, body);
      // An API acknowledgement proves storage, not that the host applied the command.
      const reported = (result as { receipt?: { status?: string } })?.receipt?.status;
      saveAction({ ...receipt, status: reported === "delivered" ? "delivered" : reported === "uncertain" ? "uncertain" : "saved", updatedAt: new Date().toISOString() });
      return result;
    } catch (error) {
      const network = error instanceof ApiError && error.code === "NETWORK";
      saveAction({ ...receipt, status: network ? (typeof navigator !== "undefined" && navigator.onLine === false ? "queued" : "uncertain") : "rejected", error: (error as Error).message, updatedAt: new Date().toISOString() });
      if (!network || attempt >= retries) throw error;
      await pause(delayMs * (attempt + 1));
    }
  }
}
let flushing: Promise<void> | null = null;
/** Called by the existing provider; no competing poller/store is created. */
export function flushActions(): Promise<void> {
  if (flushing) return flushing;
  flushing = (async () => {
    for (const receipt of [...actionReceipts()].filter(x => ["queued", "uncertain", "saved"].includes(x.status))) {
      try {
        const status = await api<{ status?: string; delivery_status?: string; command_status?: string }>(`/remotea/actions/${receipt.id}`);
        const delivery = status.delivery_status ?? status.command_status ?? status.status;
        if (["delivered", "applied", "done", "taken"].includes(delivery ?? "")) saveAction({ ...receipt, status: "delivered", error: undefined, updatedAt: new Date().toISOString() });
        else if (delivery === "failed") saveAction({ ...receipt, status: "rejected", error: "The agent could not apply this response. Review the run before trying again." });
        else if (delivery === "uncertain") saveAction({ ...receipt, status: "uncertain" });
        else saveAction({ ...receipt, status: "saved", error: undefined });
      } catch (error) {
        if (error instanceof ApiError && ["NOT_FOUND", "ACTION_NOT_FOUND"].includes(error.code) && receipt.status !== "saved") {
          try { await postAction(receipt.url, receipt.body, { retries: 0 }); } catch { /* Receipt explains the failure. */ }
        }
      }
    }
  })().finally(() => { flushing = null; });
  return flushing;
}
// Commands issued while an assistant UI action runs are tagged so the server
// never records them as the owner's own routing evidence.
let agentAction: string | null = null;
export async function asAgent<T>(
  actionId: string | undefined,
  run: () => Promise<T>,
): Promise<T> {
  if (!actionId) return run();
  const previous = agentAction;
  agentAction = actionId;
  try {
    return await run();
  } finally {
    agentAction = previous;
  }
}
export function commandId() {
  const id = crypto.randomUUID();
  return agentAction ? `ui-agent:${agentAction.slice(0, 54)}:${id}` : id;
}
export function command<T>(tool: string, args: unknown) {
  const body = { command_id: commandId(), tool, arguments: args };
  const send = () => post<{ data: T; command_id: string }>("/commands", body);
  return { send, id: body.command_id };
}
