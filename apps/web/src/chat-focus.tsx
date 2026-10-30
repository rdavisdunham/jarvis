import { flushSync } from "react-dom";
import { MessageCircle, X } from "lucide-react";
import type { RefObject } from "react";

// Mobile browsers open the on-screen keyboard only for a focus() made synchronously inside the
// user's tap. Everything here focuses before the first await; a later re-focus only restores it.

/** Focus the chat composer if it can take input. Returns whether it now has focus. */
export function focusComposer(input: RefObject<HTMLTextAreaElement | null>) {
  const element = input.current;
  if (!element || element.disabled) return false;
  element.focus({ preventScroll: true });
  return document.activeElement === element;
}

/** New conversation: focus inside the click (unless voice owns the input), run the async reset, then
 * restore focus in case the reset re-rendered or re-enabled the composer. */
export async function startNewConversation(focus: () => unknown, reset: () => Promise<void>, voiceActive: boolean) {
  if (!voiceActive) focus();
  await reset();
  focus();
}

/** The floating Eri button. Opening commits the panel synchronously so the composer exists and can be
 * focused within the same tap. */
export function ChatLauncher({ open, listening, onOpenChange, focus }: {
  open: boolean;
  listening: boolean;
  onOpenChange: (open: boolean) => void;
  /** Called only when opening; the caller skips it while voice owns the input. */
  focus: () => unknown;
}) {
  return <button type="button" className={"chat-launcher" + (open ? " is-open" : "") + (listening ? " is-listening" : "")}
    aria-label={open ? "Close Eridani" : "Open Eridani"} aria-controls="eri-conversation" aria-expanded={open}
    onClick={() => {
      const opening = !open;
      flushSync(() => onOpenChange(opening));
      if (opening) focus();
    }}>
    <span className="chat-launcher-icon" aria-hidden="true">{open ? <X size={23}/> : <MessageCircle size={24}/>}</span>
    {!open && <span className="chat-launcher-label">Eri</span>}
    {listening && <span className="chat-live-dot" aria-label="Voice active"/>}
  </button>;
}
