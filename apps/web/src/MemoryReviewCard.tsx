import { useState } from "react";
import type { MemoryReview } from "./types";

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
  return (
    <article className="memory memory-review">
      <h3>A detail to check</h3>
      <p>{review.question}</p>
      <ul>{review.candidates.map((m) => <li key={m.id}>{m.content}</li>)}</ul>
      <form onSubmit={(event) => {
        event.preventDefault();
        if (content.trim()) void onResolve("merge", content.trim());
      }}>
        <label>
          Correct fact
          <textarea
            value={content}
            onChange={(event) => setContent(event.target.value)}
            placeholder="Write the complete fact Eri should remember."
            maxLength={20000}
            rows={2}
            required
          />
        </label>
        <button className="primary" disabled={busy || !content.trim()}>
          Save corrected memory
        </button>
      </form>
      <footer>
        <button type="button" className="text-button" disabled={busy}
          onClick={() => void onResolve("distinct")}>These are different</button>
        <button type="button" className="text-button" disabled={busy}
          onClick={() => void onResolve("defer")}>Ask next week</button>
      </footer>
    </article>
  );
}
