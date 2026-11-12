import { expect, test } from "vitest";
import { autoStartNote, type AutoStart } from "./AutoStartSettings";

const base: AutoStart = { enabled: false, targets: [], daily_cap: 5, host_cap: 2, harness: "claude", choices: [], started_today: 0 };

test("the auto-start note says what the switch does right now", () => {
  expect(autoStartNote(base)).toBe("Off: every run is started by you.");
  expect(autoStartNote({ ...base, enabled: true })).toContain("nothing starts");
  expect(autoStartNote({ ...base, enabled: true, targets: ["a"], started_today: 2 })).toBe("On for 1 repository. 2 of 5 started in the last 24 hours.");
  expect(autoStartNote({ ...base, enabled: true, targets: ["a", "b"] })).toContain("2 repositories");
});
