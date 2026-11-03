import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { VoiceCoordinator } from "./voice-coordinator";
import type { CaptureActivity } from "./voice-capture";
class CaptureFake {
  latest:CaptureActivity|null=null;stream=null;microphone=null;
  open=vi.fn(async()=>{});close=vi.fn();reset=vi.fn();confirm=vi.fn();pause=vi.fn();command=vi.fn();
  listeners=new Set<(a:CaptureActivity)=>void>();
  subscribe=(fn:(a:CaptureActivity)=>void)=>{this.listeners.add(fn);return ()=>this.listeners.delete(fn);};
  flow(){this.latest={type:"activity",generation:0,active:false,epoch:0,captured:128,delivered:0,sent:0,backlog:0,startupDrained:false,level:0};for(const l of this.listeners)l(this.latest);}
}
function harness(account="a",storage=new Map<string,string>()) {
 const capture=new CaptureFake(),status=vi.fn(),start=vi.fn(async()=>{}),cancel=vi.fn(async()=>{});
 let failCapture:(message:string)=>void=()=>{};
 let options:any,wake:(r:string)=>void=()=>{},visible=true;
 const detector={start:vi.fn(),stop:vi.fn()};
 const coordinator=new VoiceCoordinator(account,"device",{capture:failed=>{failCapture=failed;return capture;},detector:(fn,o)=>{wake=fn;options=o;return detector;},start,cancel,status,
   visible:()=>visible,storage:{getItem:k=>storage.get(k)??null,setItem:(k,v)=>{storage.set(k,v);}}});
 return {coordinator,capture,status,start,cancel,detector,storage,get options(){return options;},failCapture:(message:string)=>failCapture(message),wake:(r="")=>wake(r),hide:()=>{visible=false;coordinator.suspend();},show:()=>{visible=true;return coordinator.resume(true);}};
}
beforeEach(()=>{vi.useFakeTimers();});afterEach(()=>vi.useRealTimers());
test("no provider before confirmed wake; runtime readiness needs capture AND recognition",async()=>{
 const h=harness();await h.coordinator.enable(true,true);
 expect(h.coordinator.desired).toBe(true);expect(h.start).not.toHaveBeenCalled();
 h.options.started();expect(h.coordinator.state).toBe("recovering");h.capture.flow();expect(h.coordinator.state).toBe("listening_for_wake");
 h.wake("add milk");h.wake("add milk");expect(h.start).toHaveBeenCalledOnce();expect(h.capture.confirm).toHaveBeenCalledOnce();
 expect(h.start.mock.calls[0]).toHaveLength(2);h.coordinator.dispose();
});
test("preference survives failures/reload, stays account isolated, and off releases capture",async()=>{
 const h=harness();await h.coordinator.enable(true,true);h.options.failed("network");expect(h.coordinator.desired).toBe(true);
 const same=harness("a",h.storage),other=harness("b",h.storage);
 expect(same.coordinator.desired).toBe(true);expect(other.coordinator.desired).toBe(false);
 await h.coordinator.enable(false);expect(h.capture.close).toHaveBeenCalled();
 h.coordinator.dispose();same.coordinator.dispose();other.coordinator.dispose();
});
test("permission failures never retry-loop; hidden suspension clears pending audio",async()=>{
 const h=harness();await h.coordinator.enable(true,true);h.options.failed("not-allowed");
 await vi.advanceTimersByTimeAsync(10000);expect(h.detector.start).toHaveBeenCalledOnce();expect(h.coordinator.state).toBe("permission_blocked");
 const h2=harness();await h2.coordinator.enable(true,true);h2.hide();expect(h2.capture.close).toHaveBeenCalledOnce();
 expect(h2.coordinator.desired).toBe(true);expect(h2.coordinator.state).toBe("paused_hidden");h.coordinator.dispose();h2.coordinator.dispose();
});
test("old recognition/capture promises cannot create sessions after account disposal",async()=>{
 const h=harness();await h.coordinator.enable(true,true);const old=h.options;h.coordinator.dispose();old.started();h.wake();expect(h.start).not.toHaveBeenCalled();
});
test("ten wake/end cycles own one session at a time and no paid standby",async()=>{
 const h=harness();await h.coordinator.enable(true,true);
 for(let i=0;i<10;i++){h.options.started();h.capture.flow();h.wake("add milk");h.coordinator.draining(false);await h.coordinator.ended();}
 expect(h.start).toHaveBeenCalledTimes(10);expect(h.capture.confirm).toHaveBeenCalledTimes(10);expect(h.capture.reset).toHaveBeenCalledTimes(10);
 expect(h.coordinator.state).toBe("recovering"); // readiness isn't published until actual capture and recognition restart
 h.options.started();h.capture.flow();expect(h.coordinator.state).toBe("listening_for_wake");h.coordinator.dispose();
});

test("manual voice reuses prepared foreground capture and disabled wake stays off afterwards",async()=>{
 const h=harness();await h.coordinator.enable(true,true);await h.coordinator.manual();
 expect(h.capture.open).toHaveBeenCalledOnce();expect(h.start).toHaveBeenCalledOnce();
 h.coordinator.draining(false);await h.coordinator.enable(false);
 expect(h.capture.close).not.toHaveBeenCalled();await h.coordinator.ended();expect(h.capture.close).toHaveBeenCalledOnce();
 expect(h.coordinator.state).toBe("off");h.coordinator.dispose();
});
test("wake-off while startup is pending cancels it; deadline never retries uncertain audio",async()=>{
 const h=harness();await h.coordinator.enable(true,true);h.wake("add milk");await h.coordinator.enable(false);
 expect(h.cancel).toHaveBeenCalled();expect(h.capture.close).toHaveBeenCalled();await vi.advanceTimersByTimeAsync(60000);expect(h.start).toHaveBeenCalledOnce();
 h.coordinator.dispose();
 const next=harness();await next.coordinator.enable(true,true);next.wake("add milk");await vi.advanceTimersByTimeAsync(30000);
 expect(next.coordinator.state).toBe("needs_gesture");expect(next.start).toHaveBeenCalledOnce();expect(next.cancel).toHaveBeenCalled();next.coordinator.dispose();
});

test("hidden/visible and online-style resume do not repeat a denied permission prompt",async()=>{
 const h=harness();await h.coordinator.enable(true,true);h.options.failed("not-allowed");h.hide();await h.coordinator.resume();
 expect(h.capture.open).toHaveBeenCalledOnce();expect(h.coordinator.state).toBe("permission_blocked");h.coordinator.dispose();
});

test("manual end releases a manual microphone before graceful provider settlement",async()=>{
 const h=harness();await h.coordinator.manual();h.coordinator.draining(false);h.coordinator.localEnd();
 expect(h.capture.close).toHaveBeenCalledOnce();expect(h.coordinator.state).toBe("recovering");h.coordinator.dispose();
});

test("capture interruption remains owned after several wake/voice/end generations",async()=>{
 const h=harness();await h.coordinator.enable(true,true);
 for(let i=0;i<3;i++){h.options.started();h.capture.flow();h.wake();h.coordinator.draining(false);await h.coordinator.ended();}
 h.failCapture("Microphone disconnected");expect(h.capture.close).toHaveBeenCalled();expect(h.coordinator.state).toBe("needs_gesture");h.coordinator.dispose();
});

test("storage failure keeps runtime preference truthful and safe",async()=>{
 const capture=new CaptureFake(),status=vi.fn();
 const c=new VoiceCoordinator("a","d",{capture:()=>capture,detector:()=>({start(){},stop(){}}),start:async()=>{},cancel:async()=>{},status,visible:()=>true,
   storage:{getItem(){throw Error("blocked");},setItem(){throw Error("blocked");}}});
 expect(c.desired).toBe(false);await c.enable(true,true);expect(c.desired).toBe(true);expect(c.preferenceSaved).toBe(false);
 expect(status.mock.calls.at(-1)?.[1]).toMatch(/visit only/);c.dispose();
});

test("rapid off/on abandons an unresolved capture without blocking the next generation",async()=>{
 let resolve!:()=>void;const first=new CaptureFake(),second=new CaptureFake();first.open=vi.fn(()=>new Promise<void>(r=>{resolve=r;}));
 const captures=[first,second];const c=new VoiceCoordinator("a","d",{capture:()=>captures.shift()!,detector:()=>({start(){},stop(){}}),start:async()=>{},cancel:async()=>{},status:vi.fn(),visible:()=>true,storage:{getItem:()=>null,setItem(){}}});
 const pending=c.enable(true,true);await c.enable(false);await c.enable(true,true);
 expect(second.open).toHaveBeenCalledOnce();resolve();await pending;expect(second.close).not.toHaveBeenCalled();await c.dispose();
});
