import { useState } from "react";
import { X } from "lucide-react";
import { Dialog } from "./ux";
import { SETUP_MODE_QUESTION, type SetupMode } from "./setup-model";

/** "Update with Dani" outside the setup panel (Memory → About you): the same either/or mode question. */
export function SetupModeDialog({ title, voiceAvailable, onStart, onClose }: {
  title: string; voiceAvailable: boolean; onStart: (mode: SetupMode) => Promise<void>; onClose: () => void;
}) {
  const [busy, setBusy] = useState(false), [error, setError] = useState("");
  const start = async (mode: SetupMode) => {
    setBusy(true); setError("");
    try { await onStart(mode); }
    catch (e) { setError(e instanceof Error ? e.message : "Could not start setup. Try again."); }
    finally { setBusy(false); }
  };
  return <Dialog className="dialog setup-mode-dialog" aria-label={title} onEscape={busy ? undefined : onClose}>
    <header className="dialog-heading"><h2>{title}</h2>
      <button className="btn-icon" aria-label="Close" disabled={busy} onClick={onClose}><X size={20}/></button></header>
    {error && <p className="error-banner" role="alert">{error}</p>}
    <p>{SETUP_MODE_QUESTION}</p>
    {!voiceAvailable && <p className="footnote">Voice is unavailable right now, so this will be typed.</p>}
    <div className="dialog-actions">
      <button className="btn" disabled={busy} onClick={onClose}>Back</button>
      <button className="btn" disabled={busy} onClick={() => void start("typed")}>Type instead</button>
      <button className="btn btn-primary" disabled={busy || !voiceAvailable} onClick={() => void start("voice")}>Use voice mode</button>
    </div>
  </Dialog>;
}

/** Finishing with conversation history off: Dani kept the setup transcript, so offer to delete it (decision 7). */
export function SetupFinishDialog({ onFinish, onClose }: {
  onFinish: (deleteConversation: boolean) => Promise<void>; onClose: () => void;
}) {
  const [busy, setBusy] = useState(false), [error, setError] = useState("");
  const finish = async (remove: boolean) => {
    setBusy(true); setError("");
    try { await onFinish(remove); }
    catch (e) { setError(e instanceof Error ? e.message : "Could not finish setup. Try again."); setBusy(false); }
  };
  return <Dialog className="dialog" aria-label="Finish setup" onEscape={busy ? undefined : onClose}>
    <header className="dialog-heading"><h2>Finish setup</h2>
      <button className="btn-icon" aria-label="Close" disabled={busy} onClick={onClose}><X size={20}/></button></header>
    {error && <p className="error-banner" role="alert">{error}</p>}
    <p>Conversation history is off, but Dani kept this setup conversation to build your proposal. Delete it now? What you created and saved stays.</p>
    <div className="dialog-actions">
      <button className="btn" disabled={busy} onClick={() => void finish(false)}>Keep it</button>
      <button className="btn btn-danger" disabled={busy} onClick={() => void finish(true)}>Delete conversation</button>
    </div>
  </Dialog>;
}
