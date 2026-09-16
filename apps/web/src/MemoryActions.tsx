import { useEffect, useState } from "react";
import { X } from "lucide-react";
import { api } from "./api";
import { useDialogFocus } from "./components";
import type { Memory } from "./types";
export function MemoryActions({memory, onCorrect, onForget}: {memory:Memory; onCorrect:()=>void; onForget:(deleteSource:boolean)=>Promise<unknown>}) {
  const [mode, setMode] = useState<"source"|"forget"|null>(null);
  return <><button className="text-button" onClick={() => setMode("source")}>View source</button><button className="text-button" onClick={onCorrect}>Correct</button><button className="text-button danger" onClick={() => setMode("forget")}>Forget</button>
    {mode && <MemoryActionDialog memory={memory} mode={mode} onClose={() => setMode(null)} onForget={onForget}/>}</>;
}
function MemoryActionDialog({memory, mode, onClose, onForget}: {memory:Memory; mode:"source"|"forget"; onClose:()=>void; onForget:(deleteSource:boolean)=>Promise<unknown>}) {
  useDialogFocus();
  const [source, setSource] = useState<{content:string; created_at:string}|null>(null), [error,setError] = useState("");
  const [deleteSource,setDeleteSource] = useState(false), [busy,setBusy] = useState(false);
  useEffect(() => {let live = true; if (mode === "source") void api<{content:string;created_at:string}>("/sources/" + memory.source_id).then(v => {if(live)setSource(v);}).catch(e => {if(live)setError(e.message);}); return () => {live = false;};}, [mode, memory.source_id]);
  useEffect(() => {const escape = (event:KeyboardEvent) => {if(event.key === "Escape") {event.stopImmediatePropagation();if(!busy)onClose();}};document.addEventListener("keydown",escape,true);return () => document.removeEventListener("keydown",escape,true);}, [busy,onClose]);
  return <div className="modal-backdrop"><section role="dialog" aria-modal="true" aria-label={mode === "source" ? "Memory source" : "Forget memory"} className="dialog memory-source">
    <header className="dialog-heading"><h2>{mode === "source" ? "Memory source" : "Forget this memory?"}</h2><button autoFocus className="icon-button" aria-label="Close memory action" disabled={busy} onClick={onClose}><X size={20}/></button></header>
    {error && <p role="alert">{error}</p>}
    {mode === "source" ? <>{source ? <><small>{new Date(source.created_at).toLocaleString()}</small><p className="source-content">{source.content}</p></> : !error && <p role="status">Loading source…</p>}<p className="footnote">Correct updates the remembered fact. It does not rewrite what you originally said.</p></> : <><blockquote>{memory.content}</blockquote><p>Eri will stop using this fact. Your tasks and notes stay as they are.</p><label className="inline-check"><input type="checkbox" checked={deleteSource} onChange={e => setDeleteSource(e.target.checked)}/>Also delete the stored source and every memory learned from it</label><p className="footnote">This cannot be undone. Leaving the box unchecked keeps the source and other learned facts.</p><button className="secondary" disabled={busy} onClick={onClose}>Keep memory</button><button className="primary" disabled={busy} onClick={async () => {setBusy(true);try {const result = await onForget(deleteSource);if(result)onClose();else setError("Could not forget this memory. Try again.");} catch(e) {setError((e as Error).message);} finally {setBusy(false);}}}>{busy ? "Forgetting…" : "Forget memory"}</button></>}
  </section></div>;
}
