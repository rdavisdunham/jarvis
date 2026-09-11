import { afterEach, expect, test, vi } from "vitest";
import { LiveTranscript } from "./live-transcript";
import { isWakePhrase } from "./wake-word";

afterEach(() => {
  vi.useRealTimers();
});
test("overlapping captions keep each speaker and deduplicate fragments", () => {
  vi.useFakeTimers();
  const changed = vi.fn();
  const captions = new LiveTranscript("live:", changed);
  const a = {
    type: "session.input_transcript.delta",
    event_id: "a",
    delta: "Can you",
    start_ms: 0,
    end_ms: 500,
  };
  captions.receive(a);
  captions.receive(a);
  captions.receive({
    type: "session.output_transcript.delta",
    event_id: "b",
    delta: "Of course.",
    start_ms: 100,
    end_ms: 600,
  });
  captions.receive({
    type: "session.input_transcript.delta",
    event_id: "c",
    delta: " help?",
    start_ms: 550,
    end_ms: 1000,
  });
  expect(changed).toHaveBeenCalledTimes(3);
  expect(changed.mock.calls[0][0].id).toBe(changed.mock.calls[2][0].id);
  expect(changed.mock.calls[2][0].content).toBe("Can you help?");
  vi.advanceTimersByTime(2000);
  expect(changed.mock.calls.at(-1)?.[0].pending).toBe(false);
  captions.dispose();
});
test("a long pause creates a new display bubble, not a tool event", () => {
  const changed = vi.fn();
  const captions = new LiveTranscript("live:", changed);
  captions.receive({
    type: "session.input_transcript.delta",
    delta: "First.",
    start_ms: 0,
    end_ms: 500,
  });
  captions.receive({
    type: "session.input_transcript.delta",
    delta: "Second.",
    start_ms: 4000,
    end_ms: 5000,
  });
  expect(changed.mock.calls.at(-1)?.[0].id).not.toBe(
    changed.mock.calls[0][0].id,
  );
  captions.dispose();
});
test("wake phrase is explicit, not a mention in ordinary conversation", () => {
  expect(isWakePhrase("Hey, Eri!")).toBe(true);
  expect(isWakePhrase("hey Erie")).toBe(true);
  expect(isWakePhrase("Eri")).toBe(true);
  expect(isWakePhrase("Eridani!")).toBe(true);
  expect(isWakePhrase("I wonder what Eri thinks")).toBe(false);
  expect(isWakePhrase("I told her hey Eri yesterday")).toBe(false);
  expect(isWakePhrase("very interesting")).toBe(false);
  expect(isWakePhrase("Hey Eri delete everything")).toBe(false);
});

test("large assistant chunks reveal words while input captions remain immediate", () => {
  vi.useFakeTimers();
  const changed = vi.fn();
  const captions = new LiveTranscript("live:", changed);
  const text = "Certainly, I have saved your task for tomorrow morning.";
  captions.receive({
    type: "session.output_transcript.delta",
    delta: text,
    start_ms: 0,
    end_ms: 2400,
  });
  expect(changed.mock.calls[0][0].content).not.toBe(text);
  vi.advanceTimersByTime(240);
  expect(changed.mock.calls.at(-1)?.[0].content.length).toBeGreaterThan(10);
  expect(changed.mock.calls.at(-1)?.[0].content).not.toBe(text);
  vi.advanceTimersByTime(1800);
  expect(changed.mock.calls.at(-1)?.[0].content).toBe(text);
  expect(changed.mock.calls.at(-1)?.[0].pending).toBe(false);
  captions.dispose();
});

test("closing context receives raw assistant text before animated captions catch up", () => {
  vi.useFakeTimers();
  const display = vi.fn(),
    observed = vi.fn();
  const captions = new LiveTranscript("live:", display, observed);
  const text = "The task is saved. Will that be all for now?";
  captions.receive({
    type: "session.output_transcript.delta",
    event_id: "a",
    delta: text,
    start_ms: 0,
    end_ms: 2000,
  });
  expect(observed).toHaveBeenCalledWith(
    expect.objectContaining({ content: text, pending: true }),
  );
  expect(display.mock.calls[0][0].content).not.toBe(text);
  captions.dispose();
});
