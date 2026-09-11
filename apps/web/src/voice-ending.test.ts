import { expect, test } from "vitest";
import { isVoiceEnding } from "./voice-ending";

test.each(["Goodbye.", "Thank you, Eri!", "Eri, that's all.", "Okay, bye", "Thanks", "End the voice session"])("recognizes a standalone ending: %s", (text) => {
  expect(isVoiceEnding(text)).toBe(true);
});
test.each(["Thank you, now move it to Friday", "How do you say goodbye in French?", "Don't stop listening", "Thanks for saving that, but change its date", "Eri", "Hey Eri"])("keeps requests and wake phrases active: %s", (text) => {
  expect(isVoiceEnding(text)).toBe(false);
});
