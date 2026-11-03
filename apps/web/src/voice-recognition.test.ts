import { expect, test, vi } from "vitest";
import { startRecognition } from "./voice-recognition";
test("does not infer shared-track support from constructor or start arity", () => {
  const start = vi.fn((_track?: MediaStreamTrack) => {});
  const track = {kind:"audio", readyState:"live"} as MediaStreamTrack;
  expect(startRecognition({start}, track)).toBe("browser_microphone");
  expect(start).toHaveBeenCalledWith();
  expect(startRecognition({start}, track, true)).toBe("shared_track");
  expect(start).toHaveBeenLastCalledWith(track);
});
test("runtime capability errors remain errors", () => {
  expect(() => startRecognition({ start() { throw new Error("capture conflict"); } })).toThrow("capture conflict");
});

import { validatesAudioTrack } from "./voice-recognition";
test("functional track probe distinguishes ignored arguments from track validation and always aborts",()=>{
 const stopped=vi.fn();const ended={kind:"audio",readyState:"ended",stop:stopped};
 const track={kind:"audio",readyState:"live",clone:()=>ended} as unknown as MediaStreamTrack;
 const abort=vi.fn();class Legacy {start(_track?:MediaStreamTrack){}abort=abort;}
 expect(validatesAudioTrack(Legacy,track)).toBe(false);expect(abort).toHaveBeenCalledOnce();
 class Tracks {start(t?:MediaStreamTrack){if(t?.readyState==="ended")throw Object.assign(new Error(),{name:"InvalidStateError"});}abort=abort;}
 expect(validatesAudioTrack(Tracks,track)).toBe(true);expect(abort).toHaveBeenCalledTimes(2);expect(stopped).toHaveBeenCalledTimes(2);
});
