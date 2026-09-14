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
    throw new ApiError(
      "NETWORK",
      "Connection lost. Your request may have saved. Retry the same request to check.",
    );
  }
  let value;
  try {
    value = await response.json();
  } catch {
    throw new ApiError(
      "NETWORK",
      "Eridani is reconnecting. Retry the same request to check whether it saved.",
    );
  }
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
export function command<T>(tool: string, args: unknown) {
  const body = { command_id: crypto.randomUUID(), tool, arguments: args };
  const send = () => post<{ data: T; command_id: string }>("/commands", body);
  return { send, id: body.command_id };
}
