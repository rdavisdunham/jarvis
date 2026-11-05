import { useEffect, useState } from "react";
import { ExternalLink } from "lucide-react";
import { api } from "./api";
import { TASK_CARD_SECTIONS } from "./task-card-sections";

/** One Linear comment as the sync stores it (linear_sync.comment_row). */
export type LinearComment = {
  id: string; issue_id: string | null; parent_id: string | null; author: string; body: string; url: string | null;
  created_at: string | null; updated_at: string | null; edited_at: string | null; resolved_at: string | null;
};
type Comments = { identifier: string | null; url: string | null; comments: LinearComment[]; synced_at: string | null };

/** Top-level comments oldest first, each followed by its replies. A reply whose parent is missing stands alone. */
export function commentThreads(comments: LinearComment[]) {
  const ids = new Set(comments.map(c => c.id));
  const replies = new Map<string, LinearComment[]>();
  for (const c of comments) if (c.parent_id && ids.has(c.parent_id)) replies.set(c.parent_id, [...replies.get(c.parent_id) ?? [], c]);
  return comments.filter(c => !c.parent_id || !ids.has(c.parent_id)).map(c => ({ comment: c, replies: replies.get(c.id) ?? [] }));
}

const when = (value: string | null) => value ? new Date(value).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "";

function Comment({ c, reply = false }: { c: LinearComment; reply?: boolean }) {
  return <li className={"linear-comment" + (reply ? " is-reply" : "")}>
    <p className="linear-comment-head"><strong>{c.author}</strong>
      {c.created_at && <time dateTime={c.created_at}>{when(c.created_at)}</time>}
      {c.edited_at && <span className="footnote">edited</span>}
      {c.resolved_at && !reply && <span className="chip">Resolved</span>}
      {c.url && /^https:\/\/linear\.app\//.test(c.url) && <a href={c.url} target="_blank" rel="noreferrer" aria-label={`Open ${c.author}'s comment in Linear`}><ExternalLink size={12}/></a>}</p>
    <p className="linear-comment-body">{c.body}</p>
  </li>;
}

/** The Linear issue's comments, copied at each sync. Read-only: replies happen in Linear. */
export function LinearComments({ taskId }: { taskId: string }) {
  const [data, setData] = useState<Comments | null>(null), [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    setData(null); setError("");
    api<Comments>(`/integrations/linear/tasks/${taskId}/comments`)
      .then(d => { if (active) setData(d); })
      .catch(e => { if (active) setError((e as Error).message); });
    return () => { active = false; };
  }, [taskId]);
  const threads = data ? commentThreads(data.comments) : [];
  return <section className="detail-section linear-comments" aria-label={TASK_CARD_SECTIONS.linearComments}>
    <div className="detail-section-head"><h3>Comments from Linear</h3>{!!data?.comments.length && <span className="detail-count">{data.comments.length}</span>}</div>
    {!data && !error && <p className="detail-empty">Loading comments…</p>}
    {error && <p className="detail-empty">Comments could not load: {error}</p>}
    {data && !threads.length && <p className="detail-empty">No comments on this issue in Linear yet.</p>}
    {!!threads.length && <ol className="linear-comment-list">{threads.map(t => <li key={t.comment.id} className="linear-thread">
      <ol><Comment c={t.comment}/>{t.replies.map(r => <Comment key={r.id} c={r} reply/>)}</ol></li>)}</ol>}
    {data && <p className="footnote">Copied from Linear at each sync{data.synced_at ? `, last at ${when(data.synced_at)}` : ""}.
      {data.url && /^https:\/\/linear\.app\//.test(data.url) && <> <a href={data.url} target="_blank" rel="noreferrer">Reply in Linear</a></>}</p>}
  </section>;
}
