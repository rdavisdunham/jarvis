// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TaskArchivePreferences } from "./TaskArchivePreferences";
import { command } from "./api";

vi.mock("./api", async importOriginal => ({
  ...await importOriginal<typeof import("./api")>(), command: vi.fn(),
}));

let element: HTMLDivElement;
let root: Root;
beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  element = document.createElement("div");
  document.body.append(element);
  root = createRoot(element);
});
afterEach(async () => {
  await act(async () => root.unmount());
  element.remove();
  vi.resetAllMocks();
});

describe("automatic archive preference", () => {
  it("keeps the saved setting on failure and applies the confirmed value on retry", async () => {
    const onChanged = vi.fn();
    const send = vi.fn().mockRejectedValueOnce(new Error("Could not save"))
      .mockResolvedValueOnce({ data: { auto_archive_finished_tasks: false } });
    vi.mocked(command).mockReturnValue({ id: "archive-policy-test", send });
    await act(async () => root.render(<TaskArchivePreferences enabled canEdit shared={false} onChanged={onChanged} />));
    const toggle = element.querySelector("input")!;
    await act(async () => toggle.click());
    expect(toggle.checked).toBe(true);
    expect(element.querySelector('[role="alert"]')?.textContent).toBe("Could not save");
    expect(onChanged).not.toHaveBeenCalled();
    await act(async () => toggle.click());
    expect(command).toHaveBeenLastCalledWith("settings.update", { auto_archive_finished_tasks: false });
    expect(toggle.checked).toBe(false);
    expect(onChanged).toHaveBeenCalledTimes(1);
    expect(element.textContent).toContain("Automatic archiving is paused");
  });

  it("shows the workspace policy without offering edits to a non-owner", async () => {
    await act(async () => root.render(<TaskArchivePreferences enabled canEdit={false} shared onChanged={vi.fn()} />));
    const toggle = element.querySelector("input")!;
    expect(toggle.disabled).toBe(true);
    await act(async () => toggle.click());
    expect(command).not.toHaveBeenCalled();
    expect(element.textContent).toContain("Only its owner can change it");
  });
});
