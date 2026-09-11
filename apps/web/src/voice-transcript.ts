import type { ChatMessage } from "./types";

// Presentation only. The server still owns tools, decisions and retained history.
export class VoiceTranscript {
  private items = new Map<string, ChatMessage>();
  constructor(
    private prefix: string,
    private changed: (message: ChatMessage) => void,
  ) {}
  receive(event: Record<string, any>) {
    const item = event.item_id;
    if (typeof item !== "string") return;
    const id = this.prefix + item;
    const current = this.items.get(id);
    if (event.type === "input_audio_buffer.speech_started") {
      if (!current)
        this.update({ id, role: "user", content: "", pending: true });
    } else if (
      event.type === "conversation.item.input_audio_transcription.delta"
    ) {
      if (current && !current.pending) return;
      this.update({
        id,
        role: "user",
        content: (current?.content ?? "") + (event.delta ?? ""),
        pending: true,
      });
    } else if (
      event.type === "conversation.item.input_audio_transcription.completed"
    ) {
      this.update({
        id,
        role: "user",
        content: event.transcript || "(No speech recognized)",
        pending: false,
      });
    } else if (
      event.type === "conversation.item.input_audio_transcription.failed"
    ) {
      this.update({
        id,
        role: "user",
        content: current?.content || "(Transcript unavailable for this turn)",
        pending: false,
      });
    }
  }
  private update(message: ChatMessage) {
    this.items.set(message.id, message);
    this.changed(message);
  }
}
