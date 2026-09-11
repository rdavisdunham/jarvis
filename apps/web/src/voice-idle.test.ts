import { expect, test } from "vitest";
import { VoiceIdle } from "./voice-idle";
test("ends after fifteen quiet seconds, and speech near the deadline resets it", () => {
  const idle = new VoiceIdle();
  idle.start(0);
  expect(idle.expired(14999)).toBe(false);
  idle.touch(14999);
  expect(idle.expired(15000)).toBe(false);
  expect(idle.expired(29999)).toBe(true);
});
test("never expires during speech or task work and gives a full window afterwards", () => {
  const idle = new VoiceIdle();
  idle.start(0);
  idle.speech(true, 14000);
  expect(idle.expired(40000)).toBe(false);
  idle.speech(false, 40000);
  idle.state("working", 40000);
  expect(idle.expired(120000)).toBe(false);
  idle.state("listening", 120000);
  expect(idle.expired(134999)).toBe(false);
  expect(idle.expired(135000)).toBe(true);
});
test("audio playout activity extends the deadline beyond early response completion", () => {
  const idle = new VoiceIdle();
  idle.start(0);
  idle.state("speaking", 1000);
  idle.state("listening", 2000);
  for (let t = 2000; t <= 22000; t += 200) idle.touch(t);
  expect(idle.expired(36999)).toBe(false);
  expect(idle.expired(37000)).toBe(true);
});
