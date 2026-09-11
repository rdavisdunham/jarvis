import { expect, test } from "vitest";
import { isVoiceEnding, VoiceEnding, closingQuestion } from "./voice-ending";

test.each([
  "Goodbye.",
  "Thank you, Eri!",
  "Eri, that's all.",
  "Okay, bye",
  "Thanks",
  "End the voice session",
])("recognizes a standalone ending: %s", (text) => {
  expect(isVoiceEnding(text)).toBe(true);
});
test.each([
  "Thank you, now move it to Friday",
  "How do you say goodbye in French?",
  "Don't stop listening",
  "Thanks for saving that, but change its date",
  "Eri",
  "Hey Eri",
])("keeps requests and wake phrases active: %s", (text) => {
  expect(isVoiceEnding(text)).toBe(false);
});

test.each([
  ["Anything else?", "No.", true],
  ["Anything else I can help you with?", "No, I'm good.", true],
  ["Is there anything else?", "Nope, thank you.", true],
  ["Can I help you with anything else?", "Not right now.", true],
  ["Will that be all?", "Yes.", true],
  ["Will that be all for now?", "Yep, thanks.", true],
  ["Is that everything for today?", "Yeah, I'm done for now.", true],
  ["Are we done for now?", "That's right.", true],
  ["Anything else?", "Yes.", false],
  ["Anything else?", "Yes, thank you.", false],
  ["Will that be all?", "No, thank you.", false],
  ["Anything else?", "Yes, move the task.", false],
  ["Will that be all?", "No.", false],
  ["Will that be all?", "Yes, but move the task first.", false],
  ["Anything else?", "No, move the task first.", false],
  ["Would you like that reminder tomorrow?", "Yes.", false],
  ["Are you free?", "No.", false],
  ["Did you mean Friday? Anything else?", "No.", false],
  ['The phrase is "Anything else?"', "No.", false],
])(
  "uses the current closing question: %s / %s",
  (question, reply, expected) => {
    const ending = new VoiceEnding();
    ending.assistant("a", "Saved. " + question, 0);
    expect(
      ending.user(
        { id: "u", role: "user", content: reply, pending: false },
        1000,
      ),
    ).toBe(expected);
  },
);

test("an unrelated reply consumes the closing offer and an old yes/no cannot end another turn", () => {
  const ending = new VoiceEnding();
  ending.assistant("a", "Anything else?", 0);
  expect(
    ending.user(
      { id: "first", role: "user", content: "Check tomorrow.", pending: false },
      1000,
    ),
  ).toBe(false);
  // An unchanged status poll must not re-arm the old question.
  ending.assistant("a", "Anything else?", 1500);
  expect(
    ending.user(
      { id: "second", role: "user", content: "No.", pending: false },
      2000,
    ),
  ).toBe(false);
});

test("expired offers and a fresh session do not interpret an ordinary yes/no as goodbye", () => {
  const ending = new VoiceEnding();
  ending.assistant("a", "Will that be all?", 0);
  expect(
    ending.user(
      { id: "u", role: "user", content: "Yes.", pending: false },
      30001,
    ),
  ).toBe(false);
  expect(
    new VoiceEnding().user({
      id: "u",
      role: "user",
      content: "No.",
      pending: false,
    }),
  ).toBe(false);
});

test("partial confirmations wait for a complete utterance and an added request keeps listening", () => {
  const ending = new VoiceEnding();
  ending.assistant("a", "Anything else?", 0);
  expect(
    ending.user({ id: "u", role: "user", content: "No", pending: true }, 10),
  ).toBe(false);
  expect(
    ending.user(
      {
        id: "u",
        role: "user",
        content: "No, show tomorrow first.",
        pending: false,
      },
      1000,
    ),
  ).toBe(false);
});

test("a fast assistant reply cannot replace the question that a user is answering", () => {
  const ending = new VoiceEnding();
  ending.assistant("a", "Anything else?", 0);
  ending.user({ id: "u", role: "user", content: "No", pending: true }, 10);
  ending.assistant("b", "Very good.", 20);
  expect(
    ending.user(
      { id: "u", role: "user", content: "No.", pending: false },
      1000,
    ),
  ).toBe(true);
});

test("a Live reply can share a display bubble with the previous request", () => {
  const ending = new VoiceEnding();
  ending.user(
    { id: "u", role: "user", content: "Show tomorrow.", pending: false },
    0,
  );
  ending.assistant("a", "Here it is. Will that be all?", 100);
  expect(
    ending.user(
      { id: "u", role: "user", content: "Show tomorrow. Yes.", pending: false },
      200,
    ),
  ).toBe(true);
});

test.each([
  "What's next?",
  "Anything else? I'll add a task.",
  'Say "Will that be all?"',
])(
  "does not treat an ordinary or quoted question as a closing offer: %s",
  (text) => expect(closingQuestion(text)).toBeNull(),
);

test("earlier questions in one Live display bubble do not hide a later closing offer", () => {
  const ending = new VoiceEnding();
  ending.assistant("a", "Should I show tomorrow?", 0);
  expect(
    ending.user(
      { id: "u", role: "user", content: "Yes.", pending: false },
      100,
    ),
  ).toBe(false);
  ending.assistant(
    "a",
    "Should I show tomorrow? Here it is. Will that be all?",
    200,
  );
  expect(
    ending.user(
      { id: "u", role: "user", content: "Yes. Yes.", pending: false },
      300,
    ),
  ).toBe(true);
});
