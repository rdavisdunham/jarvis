import { expect, test } from "vitest";
import cases from "../../../tests/fixtures/assistant-wake.json";
import { isPartialWakePhrase, wakeRequest, WAKE_HINTS } from "./assistant-identity";

test.each(cases)("wake grammar shared with the server: $text", ({text, request}) => {
  expect(wakeRequest(text)).toBe(request);
});
test.each(["", "D", "Dan", "Donn", "Hey, Da", "okay don", "hi Dann"])("partial wake: %s", text => {
  expect(isPartialWakePhrase(text)).toBe(true);
});
test.each(["Eri", "Hey Eri", "Eridani", "Daniel", "Hey Dani add milk"])("not an incomplete wake: %s", text => {
  expect(isPartialWakePhrase(text)).toBe(false);
});
test("recognition hints cover both pronunciations and exclude retired names", () => {
  expect(WAKE_HINTS).toContain("hey danny");
  expect(WAKE_HINTS).toContain("hey donnie");
  expect(WAKE_HINTS.every(hint => wakeRequest(hint) === "")).toBe(true);
});
