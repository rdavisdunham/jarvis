import { Check, Mic } from "lucide-react";
import { SETUP_TOPICS, type SetupSession } from "./setup-model";

/** The companion's setup strip: topic chips, "Propose my organization" and, in typed mode, "Switch to voice". */
export function SetupConversation({ session, voiceActive, voiceAvailable, busy, onSwitchToVoice, onPropose, onReview, onReviewMemories }: {
  session: SetupSession; voiceActive: boolean; voiceAvailable: boolean; busy: boolean;
  onSwitchToVoice: () => void;
  /** Absent until the backend can build a proposal from the interview. */
  onPropose?: () => void;
  /** Reopens a proposed organization that is waiting for review. */
  onReview?: () => void;
  /** Reopens the suggested About you facts that are waiting for the owner. */
  onReviewMemories?: () => void;
}) {
  const covered = new Set(session.notes?.covered ?? []);
  const proposing = session.status === "proposing";
  return <div className="setup-conversation" aria-label="Setup progress" role="group">
    <ul className="setup-topics" aria-label="Setup topics">
      {SETUP_TOPICS.map(topic => <li key={topic.id} className={"chip" + (covered.has(topic.id) ? " chip-done" : "")}>
        {covered.has(topic.id) && <Check size={12} aria-hidden="true"/>}
        {topic.label}{covered.has(topic.id) && <span className="sr-only"> (covered)</span>}
      </li>)}
    </ul>
    {proposing && <p className="setup-status" role="status">Dani is putting your proposal together. It opens here when it’s ready.</p>}
    <div className="setup-conversation-actions">
      {onReview && <button type="button" className="btn btn-primary" disabled={busy} onClick={onReview}>Review proposed organization</button>}
      {onReviewMemories && <button type="button" className={"btn" + (onReview ? "" : " btn-primary")} disabled={busy} onClick={onReviewMemories}>Review what to remember</button>}
      {onPropose && !proposing && <button type="button" className={"btn" + (onReview ? "" : " btn-primary")} disabled={busy} onClick={onPropose}>Propose my organization</button>}
      {!voiceActive && <button type="button" className="btn" disabled={busy || !voiceAvailable} onClick={onSwitchToVoice}
        title={voiceAvailable ? undefined : "Voice is unavailable. Check Settings."}>
        <Mic size={16} aria-hidden="true"/>Switch to voice
      </button>}
    </div>
    {session.history_off && <p className="footnote">Conversation history is off, but Dani keeps this setup conversation to build your proposal. You can delete it when setup finishes.</p>}
  </div>;
}
