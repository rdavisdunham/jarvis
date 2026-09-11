import type { ChatMessage } from "./types";

type Group = {
  message: ChatMessage;
  full: string;
  end: number;
  finishTimer?: ReturnType<typeof setTimeout>;
  revealTimer?: ReturnType<typeof setTimeout>;
};

// Live sends timed fragments, sometimes whole sentences. Reveal large fragments in words
// for display only; keep the exact provider transcript as the source of truth.
export class LiveTranscript {
  private current = new Map<string, Group>();
  private groups = new Set<Group>();
  private seen = new Set<string>();
  constructor(
    private prefix: string,
    private changed: (message: ChatMessage) => void,
    private observed: (message: ChatMessage) => void = () => {},
  ) {}
  receive(event: Record<string, any>) {
    const role =
      event.type === "session.input_transcript.delta"
        ? "user"
        : event.type === "session.output_transcript.delta"
          ? "assistant"
          : null;
    if (!role || typeof event.delta !== "string" || !event.delta) return;
    if (event.event_id && this.seen.has(event.event_id)) return;
    if (event.event_id) this.seen.add(event.event_id);
    const start = Number(event.start_ms ?? 0),
      end = Number(event.end_ms ?? start);
    let group = this.current.get(role);
    if (!group || start - group.end > 2000 || group.full.length > 3000) {
      if (group) this.finish(group);
      group = {
        message: {
          id: this.prefix + crypto.randomUUID(),
          role,
          content: "",
          pending: true,
        },
        full: "",
        end,
      };
      this.current.set(role, group);
      this.groups.add(group);
    }
    group.full += event.delta;
    group.end = Math.max(group.end, end);
    group.message = { ...group.message, pending: true };
    // Behavioral context uses the raw transcript immediately, independently of
    // word-by-word display and its delayed completion callbacks.
    this.observed({ ...group.message, content: group.full });
    clearTimeout(group.finishTimer);
    if (
      role === "user" ||
      (typeof matchMedia !== "undefined" &&
        matchMedia("(prefers-reduced-motion: reduce)").matches)
    ) {
      group.message = { ...group.message, content: group.full };
      this.changed(group.message);
    } else if (!group.revealTimer) this.reveal(group);
    const current = group;
    group.finishTimer = setTimeout(() => this.finish(current), 1800);
  }
  private reveal(group: Group) {
    const rest = group.full.slice(group.message.content.length);
    if (!rest) {
      group.revealTimer = undefined;
      return;
    }
    const chunks = rest.match(/\s*\S+\s*/g) ?? [rest];
    // Catch up within two seconds even if a large provider fragment arrives at once.
    const count = Math.max(1, Math.ceil(chunks.length / 20));
    const next = chunks.slice(0, count).join("");
    group.message = { ...group.message, content: group.message.content + next };
    this.changed(group.message);
    group.revealTimer = setTimeout(() => this.reveal(group), 80);
  }
  private finish(group: Group) {
    clearTimeout(group.finishTimer);
    clearTimeout(group.revealTimer);
    group.revealTimer = undefined;
    group.message = { ...group.message, content: group.full, pending: false };
    this.changed(group.message);
  }
  dispose() {
    this.groups.forEach((group) => this.finish(group));
    this.current.clear();
    this.groups.clear();
    this.seen.clear();
  }
}
