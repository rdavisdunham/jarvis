// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { Dialog } from "./ux";
import { DateField, type DateChange } from "./DatePicker";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let host: HTMLDivElement, root: Root;
beforeEach(() => { host = document.createElement("div"); document.body.append(host); root = createRoot(host); });
afterEach(() => { act(() => root.unmount()); host.remove(); });

const TODAY = "2026-11-11"; // a Wednesday
const panel = () => document.querySelector<HTMLElement>(".floating-panel.date-panel");
const button = (text: string) => Array.from(panel()!.querySelectorAll("button")).find(b => b.textContent?.startsWith(text))!;
function type(input: HTMLInputElement, value: string) {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, value);
  input.dispatchEvent(new Event("input", { bubbles: true }));
}
async function open(label: string) {
  await act(async () => host.querySelector<HTMLButtonElement>(`[aria-label="Change ${label}"]`)!.click());
  expect(panel()).not.toBeNull();
}

it("picks a do date from the quick picks, in a panel outside the card", async () => {
  const onChange = vi.fn<(c: DateChange) => void>(), onEscape = vi.fn();
  await act(async () => root.render(<Dialog aria-label="Task" onEscape={onEscape}>
    <DateField label="Do date" value={null} today={TODAY} empty="Not set" onChange={onChange}/>
  </Dialog>));
  expect(host.textContent).toContain("Not set");
  await open("Do date");
  expect(host.contains(panel())).toBe(false);
  expect(Array.from(panel()!.querySelectorAll(".date-quick-pick")).map(b => b.firstChild?.textContent)).toEqual(["Today", "Tomorrow", "This weekend", "Next week"]);
  // Escape closes only the panel, not the task card.
  await act(async () => { button("Today").dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true })); });
  expect(panel()).toBeNull();
  expect(onEscape).not.toHaveBeenCalled();
  await open("Do date");
  await act(async () => button("This weekend").click());
  expect(onChange).toHaveBeenCalledWith({ date: "2026-11-14" });
  expect(panel()).toBeNull();
});

it("shows a Sunday-first month, moves with the arrow keys and picks a day", async () => {
  const onChange = vi.fn<(c: DateChange) => void>();
  await act(async () => root.render(<DateField label="Do date" value="2026-11-30" today={TODAY} empty="Not set" onChange={onChange}/>));
  await open("Do date");
  const grid = panel()!.querySelector<HTMLElement>("[role=grid]")!;
  expect(grid.getAttribute("aria-label")).toBe("November 2026");
  const days = Array.from(grid.querySelectorAll<HTMLButtonElement>(".date-day"));
  expect(days[0].dataset.day).toBe("2026-11-01");
  expect(grid.querySelector("[aria-current=date]")?.getAttribute("data-day")).toBe(TODAY);
  expect(grid.querySelector("[aria-selected=true] button")?.getAttribute("data-day")).toBe("2026-11-30");
  // The selected day is the one tab stop; → crosses into December.
  const selected = grid.querySelector<HTMLButtonElement>("[data-day='2026-11-30']")!;
  expect(selected.tabIndex).toBe(0);
  await act(async () => { selected.focus(); selected.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true })); });
  expect(panel()!.querySelector("[role=grid]")!.getAttribute("aria-label")).toBe("December 2026");
  expect((document.activeElement as HTMLElement).dataset.day).toBe("2026-12-01");
  await act(async () => (document.activeElement as HTMLButtonElement).click());
  expect(onChange).toHaveBeenCalledWith({ date: "2026-12-01" });
});

it("types a far-off date and presses Enter", async () => {
  const onChange = vi.fn<(c: DateChange) => void>();
  await act(async () => root.render(<DateField label="Do date" value={null} today={TODAY} empty="Not set" onChange={onChange}/>));
  await open("Do date");
  const typed = panel()!.querySelector<HTMLInputElement>('input[aria-label="Do date"]')!;
  await act(async () => type(typed, "2030-01-15"));
  await act(async () => { typed.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true })); });
  expect(onChange).toHaveBeenCalledWith({ date: "2030-01-15" });
});

it("adds a due time with the home zone, and Clear removes date, time and zone", async () => {
  const onChange = vi.fn<(c: DateChange) => void>();
  const field = (time: string | null) => <DateField label="Due date" value="2026-11-12" today={TODAY} empty="No deadline" onChange={onChange}
    time={{ value: time, zone: time ? "America/Chicago" : null, homeZone: "America/Chicago" }}/>;
  await act(async () => root.render(field(null)));
  await open("Due date");
  expect(panel()!.querySelector('input[aria-label="Due time"]')).toBeNull();
  await act(async () => button("Add time").click());
  const clock = panel()!.querySelector<HTMLInputElement>('input[aria-label="Due time"]')!;
  await act(async () => type(clock, "14:30"));
  await act(async () => button("Set time").click());
  expect(onChange).toHaveBeenLastCalledWith({ date: "2026-11-12", time: "14:30", zone: "America/Chicago" });

  await act(async () => root.render(field("14:30:00")));
  expect(host.textContent).toContain("2:30 pm");
  await open("Due date");
  expect(panel()!.textContent).toContain("Your time zone");
  await act(async () => button("Clear date").click());
  expect(onChange).toHaveBeenLastCalledWith({ date: null, time: null, zone: null });
});
