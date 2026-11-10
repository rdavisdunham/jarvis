// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it } from "vitest";
import { Dialog, Popover } from "./ux";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let host: HTMLDivElement, root: Root;
beforeEach(() => { host = document.createElement("div"); document.body.append(host); root = createRoot(host); });
afterEach(() => { act(() => root.unmount()); host.remove(); });

function Card() {
  return <Dialog aria-label="Task">
    <div style={{ overflow: "hidden" }}>
      <Popover floating label="Choose a date" buttonClassName="prop-text" button="No date" panelClassName="date-panel">
        {close => <><input aria-label="Search"/><button onClick={close}>Pick</button></>}
      </Popover>
    </div>
  </Dialog>;
}

it("renders a floating panel in document.body, outside the card, and closes it on Escape and outside clicks", async () => {
  await act(async () => root.render(<Card/>));
  const trigger = host.querySelector<HTMLButtonElement>('[aria-label="Choose a date"]')!;
  expect(trigger.className).toBe("prop-text");
  await act(async () => trigger.click());
  const panel = document.querySelector<HTMLElement>(".floating-panel.date-panel")!;
  expect(panel).not.toBeNull();
  expect(host.contains(panel)).toBe(false);  // Portaled: no ancestor can clip it.
  expect(panel.closest(".floating-frame")).not.toBeNull();
  expect(document.activeElement).toBe(panel.querySelector("input"));  // Focus moves into the panel.
  // A click inside the portaled panel is not "outside".
  await act(async () => { panel.querySelector("input")!.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true })); });
  expect(document.querySelector(".floating-panel")).not.toBeNull();
  await act(async () => { panel.querySelector("input")!.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true })); });
  expect(document.querySelector(".floating-panel")).toBeNull();
  expect(document.activeElement).toBe(trigger);
  await act(async () => trigger.click());
  await act(async () => { document.body.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true })); });
  expect(document.querySelector(".floating-panel")).toBeNull();
});
