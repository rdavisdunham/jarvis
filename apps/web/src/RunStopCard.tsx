import { useState } from "react";
import { Check, SquareTerminal } from "lucide-react";
import { stopCard, type RunDetail } from "./remotea-model";

/** What a parked run needs from you, first thing in the drawer: the headline and action from the run's
 * attention_detail, the last lines of output, and the three ways on (stopCard). Approve asks first, with a note
 * recorded on the override; Send back takes the note the agent gets. A run whose agent exited also offers Reopen
 * terminal when the server says it can (run.reopen), shown as pending until the host answers. */
export function RunStopCard({ run, disabled, onApprove, onVerifyAgain, onSendBack, onReopen }: {
  run: RunDetail; disabled: boolean; onApprove: (note: string) => void; onVerifyAgain: () => void; onSendBack: (note: string) => void;
  onReopen?: () => void;
}) {
  const [mode, setMode] = useState<"" | "approve" | "send">(""), [note, setNote] = useState("");
  const card = stopCard(run);
  if (!card) return null;
  const close = () => { setMode(""); setNote(""); };
  return <section className="remotea-question run-decision run-stop-card" aria-label={card.manual ? "Needs your verification" : "Needs you"}>
    <p className="remotea-card-kicker">{card.manual ? "Needs your verification" : "Needs you"}</p>
    <p className="run-stop-headline"><strong>{card.headline}</strong></p>
    <p className="run-stop-action">{card.action}</p>
    {!!card.lines.length && <pre className="run-stop-output" aria-label="Last output">{card.lines.join("\n")}</pre>}
    {card.detail && <p className="footnote run-stop-detail">{card.detail}</p>}
    {!mode && <div className="remotea-control-row">
      {card.reopen && onReopen && <button className="btn btn-sm btn-primary" disabled={disabled || !card.reopen.enabled} title={card.reopen.reason}
        onClick={onReopen}><SquareTerminal size={14}/>{card.reopen.label}</button>}
      <button className={"btn btn-sm" + (card.reopen && onReopen ? "" : " btn-primary")} disabled={disabled || !card.approve.enabled}
        title={card.manual ? "You checked the work: record it as verified and complete the task" : "Accept the work as complete, without the recorded evidence"}
        onClick={() => setMode("approve")}><Check size={14}/>{card.approve.label}</button>
      <button className="btn btn-sm" disabled={disabled || !card.verifyAgain.enabled} title={card.verifyAgain.reason || "Run the verification again, then judge"}
        onClick={onVerifyAgain}>{card.verifyAgain.label}</button>
      <button className="btn btn-sm" disabled={disabled || !card.sendBack.enabled} title={card.sendBack.reason || "Send the agent a note; the run resumes"}
        onClick={() => setMode("send")}>{card.sendBack.label}</button>
    </div>}
    {mode === "approve" && <div className="run-stop-form" role="group" aria-label={card.approve.button}>
      <p>{card.approve.confirm}</p>
      <textarea aria-label="Why you approve" rows={2} placeholder={card.manual ? "What you checked (optional; recorded with the approval)" : "Why (optional; recorded with the approval)"} value={note}
        onChange={e => setNote(e.target.value)}/>
      <div className="remotea-control-row">
        <button className="btn btn-sm btn-primary" disabled={disabled} onClick={() => { onApprove(note.trim()); close(); }}><Check size={14}/>{card.approve.button}</button>
        <button className="btn btn-sm btn-ghost" onClick={close}>Cancel</button>
      </div>
    </div>}
    {mode === "send" && <div className="run-stop-form" role="group" aria-label={card.sendBack.label}>
      <textarea aria-label="Note for the agent" rows={3} placeholder={card.manual ? "What is wrong, or what to check" : "What the agent should do next"} value={note} onChange={e => setNote(e.target.value)}/>
      <div className="remotea-control-row">
        <button className="btn btn-sm btn-primary" disabled={disabled || !note.trim()} onClick={() => { onSendBack(note.trim()); close(); }}>Send back</button>
        <button className="btn btn-sm btn-ghost" onClick={close}>Cancel</button>
      </div>
    </div>}
  </section>;
}
