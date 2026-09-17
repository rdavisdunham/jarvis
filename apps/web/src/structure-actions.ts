import { useCallback, useRef, useState } from "react";
import { command, ApiError } from "./api";
export function useStructureActions(){
  const [error,setError]=useState("");const [busy,setBusy]=useState(false);
  const pending=useRef<{key:string;send:()=>Promise<{data:unknown}>}|null>(null);
  const run=useCallback(async<T,>(tool:string,args:unknown):Promise<T>=>{
    const key=JSON.stringify([tool,args]);
    setBusy(true);setError("");
    try {
      if(pending.current&&pending.current.key!==key){await pending.current.send();pending.current=null;}
      if(!pending.current)pending.current={key,send:command<unknown>(tool,args).send};
      const result=await pending.current.send();pending.current=null;return result.data as T;
    }
    catch(e){if(!(e instanceof ApiError)||e.code!=="NETWORK")pending.current=null;setError(e instanceof Error?e.message:"Could not save. Retry to check this request.");throw e;}
    finally{setBusy(false);}
  },[]);
  return {run,error,busy,setError};
}
