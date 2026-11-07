// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { RemoteaProvider, useRemotea, useRemoteaActions, type RemoteaActions, type RemoteaData } from "./remotea-context";

vi.mock("./api", () => ({ api: vi.fn(), flushActions: vi.fn(), postAction: vi.fn() }));
vi.mock("./RunWorkflow", () => ({ respondDecision: vi.fn(), flushDrafts: vi.fn() }));
vi.mock("./RunModels", () => ({ flushRolePreferences: vi.fn() }));
vi.mock("./remotea-refresh", () => ({ useAgentRefresh: vi.fn() }));

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
afterEach(() => vi.resetAllMocks());

it("coalesces overlapping run refreshes and discards results after disabling the integration", async () => {
  const pending: Array<() => void> = [];
  vi.mocked(api).mockImplementation(url => new Promise(resolve => {
    pending.push(() => resolve(url === "/remotea/status" ? { connected: true } : { items: [] }));
  }));
  let actions!: RemoteaActions;
  let data!: RemoteaData;
  function Probe() {
    actions = useRemoteaActions();
    data = useRemotea();
    return null;
  }
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  try {
    await act(async () => root.render(<RemoteaProvider><Probe/></RemoteaProvider>));
    await act(async () => actions.setEnabled(true));
    expect(api).toHaveBeenCalledTimes(4);
    await act(async () => {
      for (let index = 0; index < 20; index++) actions.refresh();
    });
    expect(api).toHaveBeenCalledTimes(4);
    await act(async () => pending.splice(0).forEach(resolve => resolve()));
    expect(api).toHaveBeenCalledTimes(8);
    expect(data.status).toEqual({ connected: true });
    await act(async () => actions.setEnabled(false));
    await act(async () => pending.splice(0).forEach(resolve => resolve()));
    expect(data.status).toBeNull();
    expect(data.activeRuns).toEqual([]);
    expect(api).toHaveBeenCalledTimes(8);
  } finally {
    await act(async () => root.unmount());
    host.remove();
  }
});
