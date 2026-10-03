import {useEffect,useState} from "react";
import {post} from "./api";
import {Dialog} from "./ux";
import {useStructureActions} from "./structure-actions";
import type {CustomRecord,Schema} from "./structure-types";
type Preview={preview_hash:string;affected_count:number;descendant_count:number;promoted_count:number;types:Record<string,number>;source_effect:string;issues:string[];providers:{title:string;provider:string}[]};
export function ContentsDialog({row,schema,operation,parentId,onClose,onDone}:{row:CustomRecord;schema:Schema;operation:"move"|"archive";parentId?:string|null;onClose:()=>void;onDone:()=>Promise<void>}){
 const [mode,setMode]=useState<"subtree"|"item">("subtree"),[preview,setPreview]=useState<Preview|null>(null),[error,setError]=useState("");
 const {run,busy}=useStructureActions();
 const args={record_id:row.id,expected_revision:row.revision,schema_revision:schema.revision,operation,mode,...(operation==="move"?{parent_id:parentId??null}:{})};
 useEffect(()=>{let live=true;setPreview(null);setError("");void post<Preview>("/structure/contents/preview",args).then(p=>live&&setPreview(p)).catch(e=>live&&setError(e.message));return()=>{live=false;};},[mode,row.id,row.revision,parentId,operation,schema.revision]);
 const apply=async()=>{try{await run("record.contents",{...args,preview_hash:preview!.preview_hash});await onDone();onClose();}catch(e){setError((e as Error).message);}};
 return <Dialog className="contents-dialog" aria-label={operation==="move"?"Move contents":"Archive contents"}>
  <h2>{operation==="move"?"Move":"Archive"} {row.title}</h2>
  <p>{operation==="archive"?"Recoverable removal from Eridani. Choose what happens to the contents.":"Choose what travels with this record."}</p>
  <label className="check-label"><input type="radio" name="contents-mode" checked={mode==="subtree"} onChange={()=>setMode("subtree")}/>{operation==="move"?"Take contents along":"Archive contents too"}</label>
  <label className="check-label"><input type="radio" name="contents-mode" checked={mode==="item"} onChange={()=>setMode("item")}/>{operation==="move"?"Move only this record":"Keep contents"} — direct children go to the old home</label>
  {preview&&<><p>{preview.affected_count} records change · {preview.descendant_count} descendants · {preview.promoted_count} children promoted</p>
   <div className="chip-row">{Object.entries(preview.types).map(([id,count])=><span className="chip" key={id}>{schema.types.find(t=>t.id===id)?.name??id}: {count}</span>)}</div>
   <p className="footnote">{preview.source_effect}</p>
   {!!preview.providers.length&&<details><summary>{preview.providers.length} connected-source records</summary>{preview.providers.map((p,i)=><p key={i}>{p.title} · {p.provider}</p>)}</details>}
   {preview.issues.map((i,n)=><p role="alert" key={n}>{i}</p>)}</>}
  {error&&<p role="alert">{error}</p>}
  <div className="settings-form-actions"><button className="btn" disabled={busy} onClick={onClose}>Cancel</button><button className="btn btn-primary" disabled={busy||!preview||!!preview.issues.length||!!error} onClick={()=>void apply()}>{operation==="move"?"Move":"Archive"}</button></div>
 </Dialog>;
}
