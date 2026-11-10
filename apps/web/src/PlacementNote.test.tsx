// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PlacementNote, type PlacementData } from "./PlacementNote";
import { api } from "./api";

vi.mock("./api", async importOriginal => ({
  ...await importOriginal<typeof import("./api")>(), api: vi.fn(),
}));

const pick = { home_id: "ti", title: "Transcript Intelligence", type_id: "project", confidence: "medium",
  reason: "QA disputes are part of call grading.", at: null };

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

async function show(data: PlacementData, canEdit = true) {
  vi.mocked(api).mockResolvedValue(data);
  const onFile = vi.fn();
  await act(async () => root.render(<PlacementNote recordId="r1" revision={1} canEdit={canEdit} onFile={onFile}/>));
  return onFile;
}

describe("PlacementNote", () => {
  it("offers a suggestion that files the task in one click", async () => {
    const onFile = await show({ suggestion: pick, filed: null, pending: false });
    expect(element.textContent).toContain("Looks like Transcript Intelligence: QA disputes are part of call grading.");
    await act(async () => (element.querySelector("button") as HTMLButtonElement).click());
    expect(onFile).toHaveBeenCalledWith("ti", "Transcript Intelligence");
  });

  it("explains an automatic filing and hides the button for viewers", async () => {
    await show({ suggestion: null, filed: { ...pick, confidence: "high" }, pending: false }, false);
    expect(element.textContent).toContain("Filed here by meaning");
    expect(element.querySelector("button")).toBeNull();
  });

  it("shows nothing when there is nothing to say", async () => {
    await show({ suggestion: null, filed: null, pending: false });
    expect(element.textContent).toBe("");
  });
});
