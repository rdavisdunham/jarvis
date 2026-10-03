import { useEffect, useState } from "react";
import { api, post } from "./api";
import { eventStreamOnline, fallbackInterval } from "./events";

export type VoiceDraft = { id: string; conversation_id: string; message: string; expires_at: string };

export function VoiceDraftCard({ draft, onResolved }: { draft: VoiceDraft; onResolved: () => void }) {
  const [message, setMessage] = useState(draft.message);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function resolve(action: "send" | "discard") {
    setBusy(true); setError("");
    try {
      await post("/voice/drafts/" + draft.id + "/" + action, action === "send" ? { message } : {});
      window.dispatchEvent(new Event("eri-work-changed"));
      onResolved();
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  return <section className="voice-draft" aria-label="Unsent voice draft">
    <strong>Unsent voice draft</strong>
    <p>Voice ended before this was sent to Eri. Kept for 24 hours.</p>
    <textarea aria-label="Recovered voice request" value={message} disabled={busy}
      onChange={e => setMessage(e.target.value)} rows={3}/>
    {message.length > 12000 && <p>Shorten this request to 12,000 characters before sending.</p>}
    {error && <p role="alert">{error}</p>}
    <div><button type="button" disabled={busy || !message.trim() || message.length > 12000}
      onClick={() => void resolve("send")}>Send</button>
      <button type="button" className="text-button" disabled={busy}
        onClick={() => void resolve("discard")}>Discard</button></div>
  </section>;
}

// Parent keys this component by account/workspace so responses and edits never cross scopes.
export function VoiceDrafts() {
  const [items, setItems] = useState<VoiceDraft[]>([]);
  const [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    let stopped = false;
    let inFlight = false;
    let again = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const load = async () => {
      if (stopped || document.hidden) return;
      if (inFlight) { again = true; return; }
      inFlight = true;
      clearTimeout(timer);
      try {
        const result = await api<{ items: VoiceDraft[] }>("/voice/drafts");
        if (!stopped) { setItems(result.items); setError(""); }
      } catch { if (!stopped) setError("Could not check for unsent voice drafts. Retrying…"); }
      finally {
        inFlight = false;
        if (!stopped) {
          if (again) { again = false; void load(); }
          else timer = setTimeout(() => void load(), fallbackInterval(eventStreamOnline(), 3000));
        }
      }
    };
    const update = () => { void load(); };
    update();
    window.addEventListener("eri-voice-drafts-changed", update);
    window.addEventListener("eri-events-status", update);
    document.addEventListener("visibilitychange", update);
    return () => {
      stopped = true; clearTimeout(timer);
      window.removeEventListener("eri-voice-drafts-changed", update);
      window.removeEventListener("eri-events-status", update);
      document.removeEventListener("visibilitychange", update);
    };
  }, [refresh]);
  return <>{error && <p className="muted" role="status">{error}</p>}
    {items.map(draft => <VoiceDraftCard key={draft.id} draft={draft}
      onResolved={() => { setItems(old => old.filter(item => item.id !== draft.id)); setRefresh(v => v + 1); }}/>)}</>;
}
