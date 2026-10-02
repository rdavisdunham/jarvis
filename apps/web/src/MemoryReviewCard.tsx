import { useState } from "react";
import { HelpCircle } from "lucide-react";
import type { MemoryReview } from "./types";
import "./notes.css";

export function MemoryReviewCard({
  review,
  busy,
  onResolve,
}: {
  review: MemoryReview;
  busy: boolean;
  onResolve: (action: "merge" | "distinct" | "defer", content?: string) => Promise<unknown>;
}) {
  const [content, setContent] = useState("");
  const id = "memory-review-" + review.id;
  return (
    <article className="panel memory-review" aria-labelledby={id}>
      <div className="panel-header">
        <HelpCircle size={18} aria-hidden className="memory-review-icon" />
        <h3 className="panel-title" id={id}>A detail to check</h3>
      </div>
      <p className="memory-review-question">{review.question}</p>
      <ul className="memory-review-candidates">
        {review.candidates.map((m) => <li key={m.id}>{m.content}</li>)}
      </ul>
      <form
        className="memory-review-merge"
        onSubmit={(event) => {
          event.preventDefault();
          if (content.trim()) void onResolve("merge", content.trim());
        }}
      >
        <label className="field">
          <span className="field-label-text">If they are the same, write the correct fact</span>
          <textarea
            aria-label="Correct fact"
            value={content}
            onChange={(event) => setContent(event.target.value)}
            placeholder="Write the complete fact Eri should remember."
            maxLength={20000}
            rows={2}
            required
          />
        </label>
        <div className="memory-review-actions">
          <button className="btn btn-primary" disabled={busy || !content.trim()}>
            Save corrected memory
          </button>
          <button type="button" className="btn" disabled={busy}
            onClick={() => void onResolve("distinct")}>These are different</button>
          <button type="button" className="btn btn-ghost" disabled={busy}
            onClick={() => void onResolve("defer")}>Ask next week</button>
        </div>
      </form>
    </article>
  );
}
