import { expect, test } from "vitest";
import { VoiceTranscript } from "./voice-transcript";
import type { ChatMessage } from "./types";

test("speech creates an ordered bubble, streams deltas, and reconciles corrected finals", () => {
  const messages = new Map<string, ChatMessage>();
  const transcript = new VoiceTranscript("session:", (m) =>
    messages.set(m.id, m),
  );
  const send = (type: string, item_id: string, extra = {}) =>
    transcript.receive({ type, item_id, ...extra });
  send("input_audio_buffer.speech_started", "first");
  send("input_audio_buffer.speech_started", "second");
  send("conversation.item.input_audio_transcription.delta", "second", {
    delta: "Later",
  });
  send("conversation.item.input_audio_transcription.delta", "first", {
    delta: "Hello ",
  });
  send("conversation.item.input_audio_transcription.delta", "first", {
    delta: "Erie",
  });
  expect(messages.get("session:first")?.content).toBe("Hello Erie");
  expect(messages.get("session:first")?.pending).toBe(true);
  send("conversation.item.input_audio_transcription.completed", "first", {
    transcript: "Hello Eri.",
  });
  send("conversation.item.input_audio_transcription.completed", "first", {
    transcript: "Hello Eri.",
  });
  send("conversation.item.input_audio_transcription.delta", "first", {
    delta: " late duplicate",
  });
  expect([...messages.keys()]).toEqual(["session:first", "session:second"]);
  expect(messages.get("session:first")).toMatchObject({
    content: "Hello Eri.",
    pending: false,
  });
});

test("internal planning and tool output do not appear as spoken conversation", () => {
  const messages: ChatMessage[] = [];
  const transcript = new VoiceTranscript("session:", (m) => messages.push(m));
  transcript.receive({
    type: "response.output_text.delta",
    item_id: "plan",
    delta: "RESPOND",
  });
  transcript.receive({
    type: "response.function_call_arguments.delta",
    item_id: "tool",
    delta: "{}",
  });
  expect(messages).toEqual([]);
});
