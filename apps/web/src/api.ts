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
/** A run action (`/remotea/runs/{id}/message`, `stop`, `approve`, `replan`, `freestyle`, `verify-config`)
 * that is safe to retry. One call gets one `action_id`; when the response is lost (`NETWORK`), the same
 * body with the same `action_id` is sent again, and the server returns the first result instead of acting
 * twice (`ACTION_MISMATCH` if the body changed). Other errors are not retried. */
export async function postAction<T>(
  url: string,
  data: Record<string, unknown> = {},
  { retries = 2, delayMs = 500 }: { retries?: number; delayMs?: number } = {},
): Promise<T> {
  const body = { ...data, action_id: crypto.randomUUID() };
  for (let attempt = 0; ; attempt++) {
    try {
      return await post<T>(url, body);
    } catch (error) {
      if (
        !(error instanceof ApiError) ||
        error.code !== "NETWORK" ||
        attempt >= retries
      )
        throw error;
      await pause(delayMs * (attempt + 1));
    }
  }
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
