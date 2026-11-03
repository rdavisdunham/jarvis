import { expect, test } from "vitest";
import corpus from "../../../tests/fixtures/voice-ending.json";
import { endingDecision, VoiceEnding, closingQuestion } from "./voice-ending";
test.each(corpus)("shared ending contract: $text / $question / $decision", scenario => {
  expect(endingDecision(scenario.text, scenario)).toBe(scenario.decision);
});
test.each(['Say "Anything else?"', 'Anything else? Will that be all?', 'Did you mean Friday?'])("unrelated/quoted question %s", text => expect(closingQuestion(text)).toBeNull());
test("raw input freshness is independent of caption grouping; continued speech cancels closure", () => {
  const policy = new VoiceEnding();
  policy.observe({type:"session.input_transcript.delta", event_id:"one", delta:"goodbye", start_ms:0, end_ms:100}, 0);
  expect(policy.explicit(999)).toBe(false);
  expect(policy.explicit(1000)).toBe(true);
  const candidate = {id:"candidate", source:{sequence:1,event_id:"one",end_ms:100}, category:"explicit"};
  expect(policy.accepts(candidate, 1000)).toBe(true);
  policy.speech(true, 1001);
  policy.speech(false, 1100);
  expect(policy.accepts(candidate, 3000)).toBe(false);
  expect(policy.explicit(3000)).toBe(false);
  policy.observe({type:"session.input_transcript.delta", event_id:"two", delta:"add milk", start_ms:2000,end_ms:3000}, 3000);
  expect(policy.accepts(candidate, 5000)).toBe(false);
  expect(policy.explicit(5000)).toBe(false);
});
test("startup backlog and duplicate provider events never authorize early closure", () => {
  const p = new VoiceEnding();
  const event = {type:"session.input_transcript.delta",event_id:"a",delta:"stop listening",end_ms:10};
  p.observe(event, 0); p.observe(event, 500);
  expect(p.explicit(1000, true)).toBe(false);
  expect(p.explicit(1000)).toBe(true);
});

test("a delayed transcript cannot adopt the epoch of newer local speech",()=>{
 const p=new VoiceEnding();p.speech(true,0);p.speech(false,100);p.speech(true,200);p.speech(false,400);
 p.observe({type:"session.input_transcript.delta",event_id:"old-bye",delta:"goodbye",start_ms:0,end_ms:100},500);
 const candidate={id:"c",source:{sequence:1,event_id:"old-bye",end_ms:100},category:"explicit"};
 expect(p.explicit(2000)).toBe(false);expect(p.accepts(candidate,2000)).toBe(false);
 p.observe({type:"session.input_transcript.delta",event_id:"new-request",delta:"add milk",start_ms:2000,end_ms:2400},2500);
 expect(p.explicit(4000)).toBe(false);
});

test("short local speech between meter messages still invalidates an ending",()=>{
 const p=new VoiceEnding();p.captured(false,1,0);
 p.observe({type:"session.input_transcript.delta",event_id:"bye",delta:"goodbye",end_ms:100},100);
 p.captured(false,2,150);expect(p.explicit(2000)).toBe(false);
});

test("server candidate TTL is honored and repeated polls never extend it",()=>{
 const p=new VoiceEnding();p.observe({type:"session.input_transcript.delta",event_id:"u",delta:"No",end_ms:100},0);
 const c={id:"c",source:{sequence:1,event_id:"u",end_ms:100},category:"confirm",expires_in_ms:100};
 expect(p.accepts(c,1000)).toBe(true);expect(p.accepts({...c,expires_in_ms:5000},1100)).toBe(false);
});
