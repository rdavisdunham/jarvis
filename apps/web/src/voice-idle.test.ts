import { expect, test } from "vitest";
import { VoiceIdle } from "./voice-idle";
test("ends after 120 quiet seconds, and speech near the deadline resets it", () => {
  const idle = new VoiceIdle();
  idle.start(0);
  expect(idle.expired(119999)).toBe(false);
  idle.touch(119999);
  expect(idle.expired(120000)).toBe(false);
  expect(idle.expired(239999)).toBe(true);
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
  expect(idle.expired(239999)).toBe(false);
  expect(idle.expired(240000)).toBe(true);
});
test("audio playout activity extends the deadline beyond early response completion", () => {
  const idle = new VoiceIdle();
  idle.start(0);
  idle.state("speaking", 1000);
  idle.state("listening", 2000);
  for (let t = 2000; t <= 22000; t += 200) idle.touch(t);
  expect(idle.expired(141999)).toBe(false);
  expect(idle.expired(142000)).toBe(true);
});
test("busy work suspends the countdown for at most two minutes", () => {
  const idle = new VoiceIdle();
  idle.start(0);
  idle.state("working", 1000);
  expect(idle.remaining(120_999)).toBe(null);
  expect(idle.remaining(121_000)).toBe(120);
  expect(idle.expired(240_999)).toBe(false);
  expect(idle.expired(241_000)).toBe(true);
});

test("fresh real activity resets the stuck-busy cap; a question gets ordinary quiet time", () => {
  const idle = new VoiceIdle(); idle.start(0); idle.state("working", 0);
  idle.touch(119000); expect(idle.remaining(120000)).toBeNull();
  expect(idle.remaining(239000)).toBe(120);
  idle.state("listening",240000); expect(idle.expired(359999)).toBe(false);
  expect(idle.expired(360000)).toBe(true);
});

test("startup audio drain starts a fresh quiet window", () => {
  const idle = new VoiceIdle(); idle.start(0); idle.backlog(true, 0);
  expect(idle.expired(300000)).toBe(false); idle.backlog(false,300000);
  expect(idle.expired(419999)).toBe(false); expect(idle.expired(420000)).toBe(true);
});

test("continuous quiet microphone samples are not activity",()=>{
 const idle=new VoiceIdle();idle.start(0);
 for(let t=100;t<120000;t+=100)idle.speech(false,t);
 expect(idle.expired(120000)).toBe(true);
});
