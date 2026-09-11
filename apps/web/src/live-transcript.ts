import type { ChatMessage } from "./types";

// Live has timed fragments, not completed turns. Group only for display.
export class LiveTranscript {
  private current = new Map<
    string,
    { message: ChatMessage; end: number; timer?: ReturnType<typeof setTimeout> }
  >();
  private seen = new Set<string>();
  constructor(
    private prefix: string,
    private changed: (message: ChatMessage) => void,
  ) {}
  receive(event: Record<string, any>) {
    const role =
      event.type === "session.input_transcript.delta"
        ? "user"
        : event.type === "session.output_transcript.delta"
          ? "assistant"
          : null;
    if (!role || typeof event.delta !== "string") return;
    if (event.event_id && this.seen.has(event.event_id)) return;
    if (event.event_id) this.seen.add(event.event_id);
    const start = Number(event.start_ms ?? 0),
      end = Number(event.end_ms ?? start);
    let group = this.current.get(role);
    if (
      !group ||
      start - group.end > 2000 ||
      group.message.content.length > 3000
    ) {
      if (group) this.finish(group);
      group = {
        message: { id: this.prefix + crypto.randomUUID(), role, content: "" },
        end,
      };
      this.current.set(role, group);
    }
    group.message = {
      ...group.message,
      content: group.message.content + event.delta,
      pending: true,
    };
    group.end = Math.max(group.end, end);
    this.changed(group.message);
    clearTimeout(group.timer);
    const current = group;
    group.timer = setTimeout(() => this.finish(current), 1400);
  }
  private finish(group: {
    message: ChatMessage;
    timer?: ReturnType<typeof setTimeout>;
  }) {
    clearTimeout(group.timer);
    group.message = { ...group.message, pending: false };
    this.changed(group.message);
  }
  dispose() {
    this.current.forEach((group) => this.finish(group));
    this.current.clear();
  }
}
