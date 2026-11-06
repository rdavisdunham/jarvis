import { useEffect, useState } from "react";
import { Sparkles } from "lucide-react";
import { api } from "./api";

/** Placement by meaning (placement.py): the client or project Eridani suggests for an unfiled task, or the one it
 * filed the task in automatically. Accepting a suggestion is an ordinary Home choice. */

type Pick = { home_id: string; title: string; type_id: string; confidence: string; reason: string; at: string | null };
export type PlacementData = { suggestion: Pick | null; filed: Pick | null; pending: boolean };

export function PlacementNote({ recordId, revision, canEdit, onFile }:
  { recordId: string; revision: number; canEdit: boolean; onFile: (homeId: string) => void }) {
  const [data, setData] = useState<PlacementData | null>(null);
  useEffect(() => {
    let live = true;
    void api<PlacementData>(`/structure/records/${recordId}/placement`).then(d => { if (live) setData(d); }).catch(() => {});
    return () => { live = false; };
  }, [recordId, revision]);
  if (!data) return null;
  if (data.suggestion) {
    const s = data.suggestion;
    return <div className="placement-note" role="status">
      <Sparkles size={13} aria-hidden="true"/>
      <span>Looks like <strong>{s.title}</strong>{s.reason ? `: ${s.reason}` : ""}</span>
      {canEdit && <button className="text-button" onClick={() => onFile(s.home_id)}>File here</button>}
    </div>;
  }
  if (data.filed) return <p className="footnote placement-note" title={data.filed.reason}>
    <Sparkles size={13} aria-hidden="true"/> Filed here by meaning{data.filed.reason ? `: ${data.filed.reason}` : ""}
  </p>;
  return null;
}
