import { useEffect, useState } from "react";

// ---- Undo toast -------------------------------------------------------------------------
export type UndoToastState = { message: string; undo?: () => Promise<unknown> } | null;
/** A six-second status toast with an optional Undo button (templates, Mark reviewed). */
export function useUndoToast(onUndone?: () => Promise<void> | void, undoneMessage = "Undone.") {
  const [toast, setToast] = useState<UndoToastState>(null), [busy, setBusy] = useState(false);
  useEffect(() => { if (!toast) return; const timer = setTimeout(() => setToast(null), 6000); return () => clearTimeout(timer); }, [toast]);
  const element = toast && <div className="toast" role="status"><span>{toast.message}</span>
    {toast.undo && <button type="button" disabled={busy} onClick={() => {
      const undo = toast.undo!; setToast(null); setBusy(true);
      void undo().then(async () => { await onUndone?.(); setToast({ message: undoneMessage }); })
        .catch(e => setToast({ message: (e as Error).message })).finally(() => setBusy(false));
    }}>Undo</button>}</div>;
  return { element, show: setToast };
}
