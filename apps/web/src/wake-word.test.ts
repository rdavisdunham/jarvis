import { expect, test } from "vitest";
import { wakeRequest } from "./wake-word";
test.each(["Eri", "Hey, Eri!", "Eridani"])("standalone wake %s", text => expect(wakeRequest(text)).toBe(""));
test.each(["Eri, add milk to my tasks", "hey Eri add milk to my tasks"])("preserves immediate request %s", text => expect(wakeRequest(text)).toBe("add milk to my tasks"));
test.each(["I told Eri yesterday", "Erin called me", "ordinary speech"])("does not wake on %s", text => expect(wakeRequest(text)).toBeNull());