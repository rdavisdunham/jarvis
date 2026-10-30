// @vitest-environment jsdom
import { act, useRef, useState } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it } from "vitest";
import { ChatLauncher, focusComposer, startNewConversation } from "./chat-focus";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement, root: Root;
beforeEach(() => {
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
});
afterEach(() => {
  act(() => root.unmount());
  host.remove();
});

const composer = () => host.querySelector("textarea");

/** The same shape as App: the panel (and its composer) only exists while open, so the launcher has to
 * commit the open state before it can focus. A New chat button resets the conversation asynchronously. */
function Shell({ voice = false, reset }: { voice?: boolean; reset: () => Promise<void> }) {
  const [open, setOpen] = useState(false);
  const input = useRef<HTMLTextAreaElement>(null);
  const focus = () => focusComposer(input);
  return <>
    {open && <aside>
      <button type="button" className="new-chat" onClick={() => void startNewConversation(focus, reset, voice)}>New chat</button>
      <textarea ref={input} aria-label="Message Eridani" disabled={voice}/>
    </aside>}
    <button type="button" className="elsewhere">Elsewhere</button>
    <ChatLauncher open={open} listening={voice} onOpenChange={setOpen} focus={() => { if (!voice) focus(); }}/>
  </>;
}

/** Click inside act() but report where focus was the moment the click handler returned: that is the
 * focus a phone browser sees within the tap, before any effect or animation frame runs. */
function click(selector: string) {
  let focused: Element | null = null;
  act(() => {
    host.querySelector<HTMLButtonElement>(selector)!.click();
    focused = document.activeElement;
  });
  return focused;
}

it("opening the panel from the Eri launcher focuses the composer within the click", () => {
  act(() => root.render(<Shell reset={async () => {}}/>));
  expect(composer()).toBeNull();
  const during = click(".chat-launcher");
  expect(composer()).not.toBeNull();
  expect(during).toBe(composer());
  expect(document.activeElement).toBe(composer());
});

it("New chat focuses the composer synchronously, before its reset finishes, and keeps it after", async () => {
  let finish = () => {};
  const reset = () => new Promise<void>(resolve => {
    finish = () => {
      // A reset that re-renders or moves focus elsewhere must not leave the cursor outside the box.
      host.querySelector<HTMLButtonElement>(".elsewhere")!.focus();
      resolve();
    };
  });
  act(() => root.render(<Shell reset={reset}/>));
  click(".chat-launcher");
  host.querySelector<HTMLButtonElement>(".elsewhere")!.focus();
  expect(document.activeElement).not.toBe(composer());
  expect(click(".new-chat")).toBe(composer());
  await act(async () => { finish(); await Promise.resolve(); });
  expect(document.activeElement).toBe(composer());
});

it("leaves focus alone while voice owns the input", () => {
  act(() => root.render(<Shell voice reset={async () => {}}/>));
  expect(click(".chat-launcher")).not.toBe(composer());
  expect(focusComposer({ current: composer() })).toBe(false);
});
