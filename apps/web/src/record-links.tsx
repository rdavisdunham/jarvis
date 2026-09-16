import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { useEditorBridge } from "./editor-control";
export const recordKinds = ["task", "note", "project", "goal", "area", "space", "actor"] as const;
export type LinkedRecord = {kind: typeof recordKinds[number]; id:string};
export function readRecordLink(search:string): (LinkedRecord & {workspace:string}) | null {
  const params = new URLSearchParams(search), [kind,id,...extra] = (params.get("record") ?? "").split(":");
  const workspace = params.get("workspace") ?? "personal";
  const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  if (!recordKinds.includes(kind as LinkedRecord["kind"]) || !uuid.test(id ?? "") || extra.length || (workspace !== "personal" && !uuid.test(workspace))) return null;
  return {kind:kind as LinkedRecord["kind"],id,workspace};
}
export function recordLink(origin:string, record:LinkedRecord, workspace:string) {
  const url = new URL("/", origin);
  url.searchParams.set("view", record.kind === "task" ? "tasks" : record.kind === "note" ? "notes" : "organize");
  if (record.kind === "task") url.searchParams.set("tab", "all");
  url.searchParams.set("record", record.kind + ":" + record.id); url.searchParams.set("workspace",workspace); return url.href;
}
const Context = createContext<{workspace:string;back:(()=>Promise<void>)|null}>({workspace:"personal",back:null});
export function RecordNavigator({children, workspace, view, onOpen}: {children:ReactNode;workspace:string;view:string;onOpen:(record:LinkedRecord)=>Promise<void>}) {
  const editor = useEditorBridge(), [trail,setTrail] = useState<LinkedRecord[]>([]), previousView = useRef(view);
  useEffect(() => {
    if (previousView.current !== view) {setTrail([]);previousView.current = view;}
    const {kind,record_id:id} = editor.summary ?? {};
    if (!id || !recordKinds.includes(kind as LinkedRecord["kind"])) return;
    setTrail(current => {
      const index = current.findIndex(r => r.kind === kind && r.id === id);
      return index >= 0 ? current.slice(0,index+1) : [...current,{kind:kind as LinkedRecord["kind"],id}].slice(-10);
    });
  }, [view, editor.summary?.kind, editor.summary?.record_id, workspace]);
  const back = trail.length > 1 ? async () => {const previous = trail[trail.length-2];await editor.act({operation:"close"});await onOpen(previous);} : null;
  return <Context.Provider value={{workspace,back}}>{children}</Context.Provider>;
}
export function RecordTools({kind,id}:LinkedRecord) {
  const context = useContext(Context), [message,setMessage] = useState("");
  return <div className="record-tools">{context.back && <button type="button" className="text-button" onClick={() => void context.back!().catch(e => setMessage(e.message))}>← Previous record</button>}
    <button type="button" className="text-button" onClick={() => void navigator.clipboard.writeText(recordLink(location.origin,{kind,id},context.workspace)).then(() => setMessage("Link copied · access is still required")).catch(() => setMessage(recordLink(location.origin,{kind,id},context.workspace)))}>Copy record link</button>
    {message && <small role="status">{message}</small>}</div>;
}
