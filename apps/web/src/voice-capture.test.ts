import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { VoiceCapture } from "./voice-capture";
import { captureError } from "./monitoring";
import { voiceCaptureFailure, voiceDiagnostics } from "./voice-timing";
import { CaptureProcessor } from "./voice-audio";
import { VoiceCoordinator } from "./voice-coordinator";

vi.mock("./monitoring",()=>({captureError:vi.fn()}));

class Worklet {
  static current:Worklet;
  port={postMessage:vi.fn(),close:vi.fn(),onmessage:null as ((event:{data:Record<string,unknown>})=>void)|null};
  onprocessorerror:(()=>void)|null=null;
  constructor(){Worklet.current=this;}
  connect(){return this;}
  disconnect=vi.fn();
  emit(data:Record<string,unknown>){this.port.onmessage?.({data:{generation:0,...data}});}
}

beforeEach(()=>{
  vi.useFakeTimers();
  const track={stop:vi.fn()};
  const stream={getTracks:()=>[track],getAudioTracks:()=>[track]};
  vi.stubGlobal("navigator",{mediaDevices:{getUserMedia:vi.fn(async()=>stream)}});
  vi.stubGlobal("AudioWorkletNode",Worklet);
  vi.stubGlobal("AudioContext",class {
    state="running";
    audioWorklet={addModule:vi.fn(async()=>{})};
    resume=vi.fn(async()=>{});
    close=vi.fn(async()=>{});
    createMediaStreamSource(){return {connect:(node:Worklet)=>node,disconnect:vi.fn()};}
    createMediaStreamDestination(){return {stream};}
  });
});
afterEach(()=>{vi.useRealTimers();vi.unstubAllGlobals();vi.clearAllMocks();});

test.each(["buffered","fresh"] as const)("confirmation waits for a %s worklet acknowledgement",async mode=>{
  const failed=vi.fn(),capture=new VoiceCapture(failed);
  await capture.open();
  const confirmation=capture.confirm();
  expect(capture.confirm()).toBe(confirmation);
  expect(Worklet.current.port.postMessage).toHaveBeenCalledWith({type:"confirm",generation:0,epoch:undefined});
  Worklet.current.emit({type:"confirmed",mode,retainedMs:2500});
  await expect(confirmation).resolves.toBe(mode);
  await vi.advanceTimersByTimeAsync(5000);
  expect(failed).not.toHaveBeenCalled();
  expect(captureError).toHaveBeenCalledTimes(mode==="fresh"?1:0);
  if(mode==="fresh")expect(voiceDiagnostics().failures.at(-1)).toEqual({reason:"continuity",generation:0,retained_ms:2500});
  capture.close();
});

test.each(["reset","pause","close"] as const)("%s rejects a pending acknowledgement",async action=>{
  const capture=new VoiceCapture(vi.fn());await capture.open();
  const confirmation=capture.confirm();
  const rejected=expect(confirmation).rejects.toThrow(/cancelled/);
  capture[action]();await rejected;
  Worklet.current.emit({type:"confirmed",mode:"fresh"});
  await vi.advanceTimersByTimeAsync(5000);
  expect(captureError).not.toHaveBeenCalled();
  capture.close();
});

test("old-generation acknowledgement cannot confirm a new capture",async()=>{
  const capture=new VoiceCapture(vi.fn());await capture.open();
  const old=capture.confirm();const cancelled=expect(old).rejects.toThrow(/cancelled/);
  capture.reset();await cancelled;
  const next=capture.confirm();const resolved=vi.fn();void next.then(resolved);
  Worklet.current.emit({type:"confirmed",mode:"fresh"});
  await vi.advanceTimersByTimeAsync(0);expect(resolved).not.toHaveBeenCalled();
  Worklet.current.emit({type:"confirmed",generation:1,mode:"buffered"});
  await expect(next).resolves.toBe("buffered");
  capture.close();
});

test("missing acknowledgement times out and records a handled capture failure",async()=>{
  const capture=new VoiceCapture(vi.fn());await capture.open();
  const confirmation=capture.confirm();
  const rejected=expect(confirmation).rejects.toThrow(/timed out/);
  await vi.advanceTimersByTimeAsync(5000);await rejected;
  expect(captureError).toHaveBeenCalledWith(expect.objectContaining({message:"Voice capture confirmation_timeout"}),{boundary:"voice_capture"});
  capture.close();
});

test.each(["overflow","processor"] as const)("%s rejects confirmation and reports a fixed content-free error",async reason=>{
  const failed=vi.fn(),capture=new VoiceCapture(failed);await capture.open();
  const confirmation=capture.confirm();const rejected=expect(confirmation).rejects.toThrow();
  if(reason==="processor")Worklet.current.onprocessorerror?.();
  else Worklet.current.emit({type:"failure",reason});
  await rejected;
  expect(failed).toHaveBeenCalledOnce();
  expect(captureError).toHaveBeenCalledWith(expect.objectContaining({message:"Voice capture "+reason}),{boundary:"voice_capture"});
  capture.close();
});

test("failure diagnostics are bounded and contain only codes and numbers",()=>{
  for(let i=0;i<40;i++)voiceCaptureFailure("continuity",i,1200);
  const failures=voiceDiagnostics().failures;
  expect(failures).toHaveLength(32);
  expect(failures[0]).toEqual({reason:"continuity",generation:8,retained_ms:1200});
  failures[0].generation=-1;
  expect(voiceDiagnostics().failures[0].generation).toBe(8);
});

test("real processor acknowledgement carries fresh recovery through capture and coordinator",async()=>{
  const start=vi.fn(async()=>{}),status=vi.fn();
  let wake:(request:string)=>void=()=>{};
  const capture=new VoiceCapture(vi.fn());
  const coordinator=new VoiceCoordinator("account","device",{
    capture:()=>capture,
    detector:(activate)=>{wake=activate;return{start(){},stop(){}};},
    start,cancel:async()=>{},status,visible:()=>true,storage:{getItem:()=>null,setItem(){}},
  });
  await coordinator.enable(true,true);
  const worklet=Worklet.current;
  const processor=new CaptureProcessor(48000,data=>{
    queueMicrotask(()=>worklet.emit(data));
  });
  worklet.port.postMessage.mockImplementation(message=>{
    queueMicrotask(()=>processor.command(message));
  });
  const frame=new Float32Array(128).fill(.3),output=new Float32Array(128);
  for(let i=0;i<500;i++)processor.process(frame,output);
  processor.command({type:"release",generation:0});
  processor.process(frame,output);
  wake("add a task");
  expect(start).not.toHaveBeenCalled();
  await vi.advanceTimersByTimeAsync(0);
  expect(start).toHaveBeenCalledWith(capture,expect.objectContaining({repeatRequest:true,wakeOnly:false}));
  expect(captureError).toHaveBeenCalledWith(expect.objectContaining({message:"Voice capture continuity"}),{boundary:"voice_capture"});
  coordinator.draining(false);
  expect(status).toHaveBeenLastCalledWith("active_voice","Dani is listening—please repeat your request.",true);
  await coordinator.dispose();
});
