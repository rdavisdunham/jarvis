// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { aliasProblem, NameAliases } from "./NameAliases";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let host: HTMLDivElement, root: Root;
beforeEach(() => { host = document.createElement("div"); document.body.append(host); root = createRoot(host); });
afterEach(() => { act(() => root.unmount()); host.remove(); });

const button = (label: string) => [...host.querySelectorAll("button")].find(b => (b.getAttribute("aria-label") ?? b.textContent ?? "").trim().startsWith(label))!;
const input = () => host.querySelector<HTMLInputElement>("input[aria-label='Add another name']")!;
function type(value: string) {
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
  act(() => { setter.call(input(), value); input().dispatchEvent(new Event("input", { bubbles: true })); });
}
const chips = () => [...host.querySelectorAll(".chip")].map(c => c.textContent);
async function render(aliases: string[], onSave = vi.fn(async () => {})) {
  await act(async () => { root.render(<NameAliases aliases={aliases} busy={false} onSave={onSave}/>); });
  return onSave;
}

describe("Also known as", () => {
  it("adds, removes and saves the whole list", async () => {
    const onSave = await render(["Davis"]);
    expect(chips()).toEqual(["Davis"]);
    expect(button("Save names").disabled).toBe(true);
    type("  Robert   Davis Dunham ");
    await act(async () => { button("Add").click(); });
    expect(chips()).toEqual(["Davis", "Robert Davis Dunham"]);
    expect(input().value).toBe("");
    await act(async () => { button("Remove Davis").click(); });
    expect(chips()).toEqual(["Robert Davis Dunham"]);
    await act(async () => { button("Save names").click(); });
    expect(onSave).toHaveBeenCalledWith(["Robert Davis Dunham"]);
  });

  it("explains a name it will not add", async () => {
    await render(["Davis Dunham"]);
    type("davis   dunham");
    await act(async () => { button("Add").click(); });
    expect(host.querySelector("[role=alert]")?.textContent).toContain("already in the list");
    expect(chips()).toEqual(["Davis Dunham"]);
    type("davis@example.com");
    expect(host.querySelector("[role=alert]")).toBeNull();
  });

  it("resets to the saved list when the profile reloads", async () => {
    await render([]);
    expect(host.textContent).toContain("No other names yet.");
    await render(["Davis Dunham"]);
    expect(chips()).toEqual(["Davis Dunham"]);
  });

  it("validates like the server", () => {
    expect(aliasProblem([], " ")).toBe("Type a name first.");
    expect(aliasProblem([], "a@b.co")).toContain("email");
    expect(aliasProblem([], "Me")).toContain("can't be used");
    expect(aliasProblem([], "x".repeat(81))).toContain("80 characters");
    expect(aliasProblem(Array.from({ length: 10 }, (_, i) => `Name ${i}`), "Another")).toContain("at most 10");
    expect(aliasProblem(["Davis"], "Davis Dunham")).toBe("");
  });
});
