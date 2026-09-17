import type { ChatMessage } from "./types";
import { workActive, workAttention, type WorkItem } from "./Activity";

export type ChatEntry = {kind: "message"; message: ChatMessage} | {kind: "work"; work: WorkItem};
export const hasSavedChanges = (item: WorkItem): boolean => !!item.actions.length || item.children.some(hasSavedChanges);
export const showChatCard = (item: WorkItem) => hasSavedChanges(item) || (!item.navigation_only && workAttention(item));
const normalized = (text: string) => text.toLocaleLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
const time = (value?: string | null) => value ? Date.parse(value) || 0 : 0;

/** Merge ordinary backend replies into the transcript, rather than using a card as the reply. */
export function mergeWorkReplies(messages: ChatMessage[], items: WorkItem[]): ChatMessage[] {
  const additions = items.filter(item => !item.voice_session_id && !workActive(item) && item.message && item.response_native_id
    && !messages.some(m => m.native_id === item.response_native_id || m.id === item.response_native_id));
  if (!additions.length) return messages;
  return [...messages, ...additions.map(item => ({id: item.response_native_id!, native_id: item.response_native_id,
    role: "assistant" as const, content: item.message, created_at: item.finished_at ?? item.updated_at}))]
    .sort((a, b) => time(a.created_at) - time(b.created_at));
}

/** Anchors are fixed on first observation, including while a job is still queued.
 * Status changes, late completions and clarification continuations never move them.
 * Native typed-turn IDs are exact; voice falls back to the original words/time.
 */
export function chatTimeline(messages: ChatMessage[], items: WorkItem[], anchors: Map<string, string | null>): ChatEntry[] {
  const positions = new Map(messages.map((m, i) => [m.id, i]));
  const buckets = new Map<number, WorkItem[]>();
  for (const item of [...items].sort((a,b) => time(a.created_at)-time(b.created_at) || a.id.localeCompare(b.id))) {
    if (!anchors.get(item.id) || !positions.has(anchors.get(item.id)!)) {
      const exact = messages.find(m => m.role === "user" && (m.id === item.id || m.native_id === `work:${item.id}:user`));
      const request = normalized(item.request);
      const candidates = messages.filter(m => m.role === "user" && (!m.created_at || time(m.created_at) <= time(item.created_at) + 1500));
      const matching = candidates.filter(m => {
        const words = normalized(m.content);
        return request.length > 3 && words.length > 3 && (words.includes(request) || request.includes(words));
      });
      const anchor = exact ?? matching.at(-1) ?? candidates.at(-1);
      anchors.set(item.id, anchor?.id ?? null);
    }
    if (!showChatCard(item)) continue;
    const index = positions.get(anchors.get(item.id) ?? "") ?? -1;
    buckets.set(index, [...(buckets.get(index) ?? []), item]);
  }
  const result: ChatEntry[] = (buckets.get(-1) ?? []).map(work => ({kind: "work", work}));
  messages.forEach((message, index) => {
    result.push({kind: "message", message});
    for (const work of buckets.get(index) ?? []) result.push({kind: "work", work});
  });
  return result;
}
