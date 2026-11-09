import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { POLL_ACTIVE_MS, POLL_HIDDEN_MS, POLL_STABLE_MS, Voice } from "./voice";
import { api, post } from "./api";
import { CaptureProcessor } from "./voice-audio";
import { VoiceCoordinator } from "./voice-coordinator";

vi.mock("./api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./api")>()),
  api: vi.fn(),
  post: vi.fn(),
}));

class Channel extends EventTarget {
  readyState = "open";
}
class Peer {
  static current: Peer;
  connectionState = "connected";
  iceGatheringState = "complete";
  onconnectionstatechange: (() => void) | null = null;
  ontrack: unknown;
  localDescription = { sdp: "synthetic" };
  constructor() {
    Peer.current = this;
  }
  addTrack() {}
  channel = new Channel();
  createDataChannel() {
    return this.channel;
  }
  async createOffer() {
    return { sdp: "synthetic" };
  }
  async setLocalDescription() {}
  async setRemoteDescription() {}
  close = vi.fn();
  change(state: string) {
    this.connectionState = state;
    this.onconnectionstatechange?.();
  }
}

let stopped = vi.fn();
beforeEach(() => {
  vi.useFakeTimers();
  stopped = vi.fn();
  const track = { enabled: true, stop: stopped };
  vi.stubGlobal("navigator", {
    mediaDevices: {
      getUserMedia: vi.fn().mockResolvedValue({
        getTracks: () => [track],
        getAudioTracks: () => [track],
      }),
    },
  });
  vi.stubGlobal("RTCPeerConnection", Peer);
  vi.stubGlobal(
    "Audio",
    class {
      autoplay = false;
      srcObject = null;
      pause() {}
    },
  );
  vi.mocked(post).mockResolvedValue({
    session_id: "test-session",
    sdp: "answer",
    ready: true,
  });
  vi.mocked(api).mockResolvedValue({
    state: "listening",
    error: null,
    text: "",
    closed: false,
    receipts: [],
  });
});
afterEach(() => {
  vi.clearAllTimers();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

test("brief disconnection recovers without ending the voice session", async () => {
  const changed = vi.fn();
  const voice = await startVoice("live", changed);
  Peer.current.change("disconnected");
  await vi.advanceTimersByTimeAsync(3000);
  expect(stopped).not.toHaveBeenCalled();
  Peer.current.change("connected");
  await vi.advanceTimersByTimeAsync(3000);
  expect(stopped).not.toHaveBeenCalled();
  expect(changed.mock.calls.some(([state]) => state.closed)).toBe(false);
  await voice.stop();
});

test("persistent disconnection releases microphone and ends the server session", async () => {
  const changed = vi.fn();
  const voice = await startVoice("live", changed);
  Peer.current.change("disconnected");
  await vi.advanceTimersByTimeAsync(5000);
  expect(stopped).toHaveBeenCalledOnce();
  expect(Peer.current.close).toHaveBeenCalledOnce();
  expect(changed).toHaveBeenCalledWith(
    expect.objectContaining({ state: "disconnected", closed: true }),
  );
  expect(api).toHaveBeenCalledWith(
    "/voice/sessions/test-session/stop",
    expect.objectContaining({ method: "POST" }),
  );
});

test("capture and listening wait for the provider data channel", async () => {
  const changed = vi.fn();
  const voice = new Voice(changed);
  let resolveSession: (value: unknown) => void = () => {};
  vi.mocked(post).mockReturnValueOnce(
    new Promise((resolve) => {
      resolveSession = resolve;
    }),
  );
  const starting = voice.start("conversation");
  await vi.advanceTimersByTimeAsync(0);
  Peer.current.channel.readyState = "connecting";
  resolveSession({ session_id: "test-session", sdp: "answer", ready: true });
  await vi.advanceTimersByTimeAsync(0);
  const stream = await vi.mocked(navigator.mediaDevices.getUserMedia).mock
    .results[0].value;
  expect(stream.getTracks()[0].enabled).toBe(false);
  expect(api).not.toHaveBeenCalled();
  Peer.current.channel.readyState = "open";
  Peer.current.channel.dispatchEvent(new Event("open"));
  await vi.advanceTimersByTimeAsync(0);
  expect(stream.getTracks()[0].enabled).toBe(false);
  event({ type: "session.started" });
  await starting;
  expect(stream.getTracks()[0].enabled).toBe(true);
  expect(api).toHaveBeenCalled();
  await voice.stop();
});

test("a late status response from a stopped session cannot update the UI", async () => {
  let resolveStatus: (value: any) => void = () => {};
  vi.mocked(api).mockReturnValueOnce(
    new Promise((resolve) => {
      resolveStatus = resolve;
    }),
  );
  const changed = vi.fn();
  const voice = await startVoice("live", changed);
  await voice.stop();
  resolveStatus({
    state: "unresolved",
    error: "stale failure",
    closed: false,
    text: "",
    receipts: [],
  });
  await vi.advanceTimersByTimeAsync(0);
  expect(changed).not.toHaveBeenCalled();
});

test("one failed status poll recovers without stopping working audio", async () => {
  const changed = vi.fn();
  vi.mocked(api).mockRejectedValueOnce(new Error("temporary 502"));
  const voice = await startVoice("live", changed);
  await vi.advanceTimersByTimeAsync(801);
  expect(stopped).not.toHaveBeenCalled();
  expect(changed).toHaveBeenCalledWith(
    expect.objectContaining({ state: "listening", error: null }),
  );
  expect(changed.mock.calls.some(([state]) => state.closed)).toBe(false);
  await voice.stop();
});

test("GPT-Live waits for session.started before enabling capture", async () => {
  const voice = new Voice(vi.fn());
  const start = voice.start("conversation", undefined, {
    provider: "live",
    voice: "willow",
  });
  await vi.advanceTimersByTimeAsync(0);
  const stream = await vi.mocked(navigator.mediaDevices.getUserMedia).mock
    .results[0].value;
  expect(stream.getTracks()[0].enabled).toBe(false);
  expect(api).not.toHaveBeenCalled();
  Peer.current.channel.dispatchEvent(
    new MessageEvent("message", {
      data: JSON.stringify({ type: "session.started" }),
    }),
  );
  await start;
  expect(stream.getTracks()[0].enabled).toBe(true);
  expect(post).toHaveBeenCalledWith(
    "/voice/sessions",
    expect.objectContaining({ provider: "live", voice: "willow" }),
  );
  await voice.stop();
});

test("GPT-Live releases the mic immediately while its peer collects final usage", async () => {
  const voice = new Voice(vi.fn());
  const start = voice.start("conversation", undefined, {
    provider: "live",
    voice: "marin",
  });
  await vi.advanceTimersByTimeAsync(0);
  Peer.current.channel.dispatchEvent(
    new MessageEvent("message", {
      data: JSON.stringify({ type: "session.started" }),
    }),
  );
  await start;
  let finish: (value: unknown) => void = () => {};
  vi.mocked(api).mockReturnValueOnce(
    new Promise((resolve) => {
      finish = resolve;
    }),
  );
  const ending = voice.stop();
  expect(stopped).toHaveBeenCalledOnce();
  expect(Peer.current.close).not.toHaveBeenCalled();
  finish({ closed: true });
  await ending;
  expect(Peer.current.close).toHaveBeenCalled();
});

test.each(["live"] as const)(
  "spoken thanks keeps %s open until manual end",
  async (provider) => {
    const changed = vi.fn();
    const voice = new Voice(changed);
    const starting = voice.start("conversation", undefined, {
      provider,
      voice: "marin",
    });
    await vi.advanceTimersByTimeAsync(0);
    if (provider === "live")
      Peer.current.channel.dispatchEvent(
        new MessageEvent("message", {
          data: JSON.stringify({ type: "session.started" }),
        }),
      );
    await starting;
    const event =
      provider === "live"
        ? {
            type: "session.input_transcript.delta",
            delta: "Thank you, Dani.",
            start_ms: 0,
            end_ms: 1000,
          }
        : {
            type: "conversation.item.input_audio_transcription.completed",
            item_id: "ending",
            transcript: "Thank you, Dani.",
          };
    Peer.current.channel.dispatchEvent(
      new MessageEvent("message", { data: JSON.stringify(event) }),
    );
    await vi.advanceTimersByTimeAsync(3000);
    expect(stopped).not.toHaveBeenCalled();
    expect(changed.mock.calls.some(([state]) => state.closed)).toBe(false);
    await voice.stop();
    const next = await startVoice("live");
    expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(2);
    await next.stop();
  },
);

test.skip("paused Realtime: a new speech start cancels a pending spoken ending", async () => {
  const voice = new Voice(vi.fn());
  await voice.start("conversation");
  Peer.current.channel.dispatchEvent(
    new MessageEvent("message", {
      data: JSON.stringify({
        type: "conversation.item.input_audio_transcription.completed",
        item_id: "thanks",
        transcript: "Thanks",
      }),
    }),
  );
  await vi.advanceTimersByTimeAsync(500);
  Peer.current.channel.dispatchEvent(
    new MessageEvent("message", {
      data: JSON.stringify({
        type: "input_audio_buffer.speech_started",
        item_id: "next",
      }),
    }),
  );
  await vi.advanceTimersByTimeAsync(700);
  expect(stopped).not.toHaveBeenCalled();
  await voice.stop();
});

function event(data: Record<string, unknown>) {
  Peer.current.channel.dispatchEvent(
    new MessageEvent("message", { data: JSON.stringify(data) }),
  );
}

async function startVoice(provider: "live" | "realtime", changed = vi.fn()) {
  const voice = new Voice(changed);
  const starting = voice.start("conversation", undefined, {
    provider,
    voice: "marin",
  });
  await vi.advanceTimersByTimeAsync(0);
  if (provider === "live") event({ type: "session.started" });
  await starting;
  return voice;
}

async function assistantQuestion(
  provider: "live" | "realtime",
  question: string,
) {
  if (provider === "live") {
    event({
      type: "session.output_transcript.delta",
      event_id: "a",
      delta: question,
      start_ms: 1000,
      end_ms: 2100,
    });
  } else {
    vi.mocked(api).mockResolvedValue({
      state: "listening",
      closed: false,
      error: null,
      receipts: [],
      text_id: "a",
      text: question,
    });
    await vi.advanceTimersByTimeAsync(400);
  }
}

function userReply(provider: "live" | "realtime", reply: string, id = "reply") {
  event(
    provider === "live"
      ? {
          type: "session.input_transcript.delta",
          event_id: id,
          delta: reply,
          start_ms: 2200,
          end_ms: 2400,
        }
      : {
          type: "conversation.item.input_audio_transcription.completed",
          item_id: id,
          transcript: reply,
        },
  );
}

test.each(["live"] as const)(
  "%s closes after a contextual confirmation and preserves provider cleanup",
  async (provider) => {
    const changed = vi.fn();
    await startVoice(provider, changed);
    await assistantQuestion(
      provider,
      "Saved. Anything else I can help you with?",
    );
    userReply(provider, "No.");
    vi.mocked(api).mockResolvedValue({state:"listening", closed:false, text:"", error:null, receipts:[],
      end_candidate:{id:"candidate",source:{sequence:1,event_id:"reply",end_ms:2400},category:"confirm"}});
    vi.mocked(post).mockResolvedValue({closed:true});
    await vi.advanceTimersByTimeAsync(3000);
    expect(stopped).toHaveBeenCalledOnce();
    expect(Peer.current.close).toHaveBeenCalledOnce();
    expect(changed).toHaveBeenCalledWith(
      expect.objectContaining({ state: "ended", closed: true }),
    );
    if (provider === "live")
      expect(api).toHaveBeenCalledWith(
        "/voice/sessions/test-session/stop",
        expect.objectContaining({ method: "POST" }),
      );
    else expect(post).toHaveBeenCalledWith("/voice/sessions/test-session/stop");
  },
);

test.each(["live"] as const)(
  "%s keeps listening when the answer requests more help",
  async (provider) => {
    const voice = await startVoice(provider);
    await assistantQuestion(provider, "Will that be all?");
    userReply(provider, "No.");
    await vi.advanceTimersByTimeAsync(3000);
    expect(stopped).not.toHaveBeenCalled();
    await voice.stop();
  },
);

test.each(["live"] as const)(
  "%s cancels an agreed ending when the user continues",
  async (provider) => {
    const voice = await startVoice(provider);
    await assistantQuestion(provider, "Will that be all?");
    userReply(provider, "Yes.");
    await vi.advanceTimersByTimeAsync(provider === "live" ? 2000 : 500);
    if (provider === "live")
      userReply(provider, " Actually, move that task first.", "continuation");
    else
      event({
        type: "input_audio_buffer.speech_started",
        item_id: "continuation",
      });
    await vi.advanceTimersByTimeAsync(1200);
    expect(stopped).not.toHaveBeenCalled();
    await voice.stop();
  },
);


test("the default idle window keeps listening for 120 seconds then releases the microphone", async () => {
  const changed = vi.fn();
  await startVoice("live", changed);
  await vi.advanceTimersByTimeAsync(119999);
  expect(stopped).not.toHaveBeenCalled();
  await vi.advanceTimersByTimeAsync(251);
  expect(stopped).toHaveBeenCalledOnce();
  expect(changed).toHaveBeenCalledWith(expect.objectContaining({state:"idle_timeout",closed:true}));
});

test("a server-side voice_end closes media and allows a fresh wake session", async () => {
  const changed = vi.fn();
  await startVoice("live", changed);
  vi.mocked(api).mockResolvedValue({state:"closed",closed:true,error:null,text:"",receipts:["saved"]});
  await vi.advanceTimersByTimeAsync(500);
  expect(stopped).toHaveBeenCalledOnce();
  expect(Peer.current.close).toHaveBeenCalledOnce();
  vi.mocked(api).mockResolvedValue({state:"listening",closed:false,error:null,text:"",receipts:[]});
  const next = await startVoice("live");
  expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(2);
  await next.stop();
});


test("queued backend work suspends idle shutdown and completion starts a fresh 120 seconds", async () => {
  const changed = vi.fn();
  vi.mocked(api).mockResolvedValue({
    state: "working", closed: false, error: null, text: "", receipts: [],
  });
  await startVoice("live", changed);
  await vi.advanceTimersByTimeAsync(90000);
  expect(stopped).not.toHaveBeenCalled();
  expect(changed.mock.calls.some(([state]) => state.closed)).toBe(false);
  vi.mocked(api).mockResolvedValue({
    state: "listening", closed: false, error: null, text: "", receipts: [],
  });
  // Stable status backs off to the slower lease cadence before noticing completion.
  await vi.advanceTimersByTimeAsync(POLL_STABLE_MS);
  await vi.advanceTimersByTimeAsync(119000);
  expect(stopped).not.toHaveBeenCalled();
  await vi.advanceTimersByTimeAsync(1200);
  expect(stopped).toHaveBeenCalledOnce();
  expect(changed).toHaveBeenCalledWith(
    expect.objectContaining({ state: "idle_timeout", closed: true }),
  );
});

test("explicit goodbye still ends voice while queued backend work is pending", async () => {
  const changed = vi.fn();
  vi.mocked(api).mockResolvedValue({
    state: "working", closed: false, error: null, text: "", receipts: [],
  });
  await startVoice("live", changed);
  await vi.advanceTimersByTimeAsync(35000);
  userReply("live", "Goodbye Dani.");
  await vi.advanceTimersByTimeAsync(3000);
  expect(stopped).toHaveBeenCalledOnce();
  expect(changed).toHaveBeenCalledWith(
    expect.objectContaining({ state: "ended", closed: true }),
  );
});

test.each([
  ["working", false],
  ["listening", true],
])(
  "a bare thanks does not hang up while backend work or its question is pending (%s)",
  async (state, awaiting) => {
    const changed = vi.fn();
    vi.mocked(api).mockResolvedValue({
      state, awaiting_answer: awaiting, closed: false, error: null, text: "", receipts: [],
    });
    const voice = await startVoice("live", changed);
    await vi.advanceTimersByTimeAsync(500);
    userReply("live", "Thanks.");
    await vi.advanceTimersByTimeAsync(3000);
    expect(stopped).not.toHaveBeenCalled();
    await voice.stop();
  },
);

test("status changes are published once and stable polling backs off", async () => {
  const changed = vi.fn();
  const voice = await startVoice("live", changed);
  await vi.advanceTimersByTimeAsync(10_000);
  // The idle countdown changes once per second; identical poll results add nothing.
  expect(changed.mock.calls.length).toBeLessThanOrEqual(12);
  const polls = vi.mocked(api).mock.calls.length;
  await vi.advanceTimersByTimeAsync(10_000);
  expect(vi.mocked(api).mock.calls.length - polls).toBeLessThanOrEqual(
    Math.ceil(10_000 / POLL_STABLE_MS) + 1,
  );
  expect(POLL_ACTIVE_MS).toBeLessThan(POLL_STABLE_MS);
  await voice.stop();
});

test("a hidden tab polls slowly but well inside the 30 second lease", async () => {
  vi.stubGlobal("document", Object.assign(new EventTarget(), { visibilityState: "hidden" }));
  const voice = await startVoice("live");
  vi.mocked(api).mockClear();
  await vi.advanceTimersByTimeAsync(20_000);
  const polls = vi.mocked(api).mock.calls.length;
  expect(polls).toBeGreaterThanOrEqual(3);
  expect(polls).toBeLessThanOrEqual(5);
  expect(POLL_HIDDEN_MS).toBeLessThanOrEqual(10_000);
  await voice.stop();
});

test.each([0,2,5,15])("prepared capture carries samples once through %s-second setup without reacquiring or text submission",async delay=>{
 const {CaptureProcessor}=await import("./voice-audio");
 const listeners=new Set<(value:any)=>void>();
 const processor=new CaptureProcessor(1000,value=>{for(const l of listeners)l(value);});
 processor.command({type:"confirm",generation:0});
 const expected:number[]=[],heard:number[]=[];
 const track={enabled:true,stop:vi.fn()};
 const capture={stream:{getTracks:()=>[track],getAudioTracks:()=>[track]},context:null,
   subscribe:(fn:(value:any)=>void)=>{listeners.add(fn);return()=>listeners.delete(fn);},
   command:(type:string,epoch?:number)=>processor.command({type,generation:0,epoch}),
   ready:vi.fn(()=>processor.command({type:"ready",generation:0})),pause:vi.fn(()=>processor.command({type:"pause",generation:0}))};
 let identity!:(id:string)=>void,ui!:(value:void)=>void;
 const id=new Promise<string>(resolve=>{identity=resolve;}),uiReady=new Promise<void>(resolve=>{ui=resolve;});
 const voice=new Voice(vi.fn());
 const start=voice.start(id,undefined,{provider:"live",voice:"marin"},capture as never,
   {id:crypto.randomUUID(),generation:1,wake:true,wakeOnly:false,detectedAt:performance.now(),epoch:1},uiReady);
 await vi.advanceTimersByTimeAsync(0);
 expect(navigator.mediaDevices.getUserMedia).not.toHaveBeenCalled();expect(post).not.toHaveBeenCalled();
 for(let i=0;i<Math.max(1,delay*10);i++){
   const input=Float32Array.from({length:100},(_,j)=>(i*100+j+1)/20000),out=new Float32Array(100);
   expected.push(...input);processor.process(input,out);expect(out.every(x=>x===0)).toBe(true);
 }
 identity("conversation");ui();await vi.advanceTimersByTimeAsync(0);
 event({type:"session.started"});await start;
 expect(capture.ready).toHaveBeenCalledOnce();
 while(processor.queue.backlog){heard.push(...processor.queue.read(Math.min(100,processor.queue.backlog)));}
 expect(heard).toEqual(expected);
 expect(vi.mocked(post).mock.calls.filter(([path])=>path==="/voice/sessions")).toHaveLength(1);
 expect(vi.mocked(post).mock.calls.some(([path])=>path==="/work")).toBe(false);
 await voice.stop();expect(track.stop).not.toHaveBeenCalled();expect(capture.pause).toHaveBeenCalledOnce();
});

test("accepted contextual close stops local audio before final provider usage settles",async()=>{
 const changed=vi.fn();await startVoice("live",changed);
 userReply("live","No.");
 vi.mocked(api).mockImplementation(async(path,options)=>{
   if(path.endsWith("/stop") && options?.method==="POST")return await new Promise(resolve=>{settle=resolve;});
   return {state:"listening",closed:false,text:"",error:null,receipts:[],end_candidate:{id:"candidate",source:{sequence:1,event_id:"reply",end_ms:2400},category:"confirm"}};
 });
 let settle!:(value:unknown)=>void;
 vi.mocked(post).mockResolvedValue({closed:false,accepted:true,status:"closing"});
 await vi.advanceTimersByTimeAsync(1200);expect(stopped).toHaveBeenCalledOnce();expect(Peer.current.close).not.toHaveBeenCalled();
 settle({closed:true});await vi.advanceTimersByTimeAsync(0);expect(Peer.current.close).toHaveBeenCalledOnce();
});

test("fresh wake recovery shows a repeat notice until the next user transcript and skips the greeting", async () => {
  const changed = vi.fn();
  const listeners = new Set<(activity: any) => void>();
  const track = { enabled: true, stop: vi.fn() };
  const capture = {
    stream: { getTracks: () => [track], getAudioTracks: () => [track] },
    context: null,
    subscribe: (listener: (activity: any) => void) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    ready: vi.fn(), pause: vi.fn(), command: vi.fn(),
  };
  const voice = new Voice(changed);
  const starting = voice.start("conversation", undefined, { provider: "live", voice: "marin" }, capture as never, {
    id: "recovered-wake", generation: 0, wake: true, wakeOnly: false,
    detectedAt: performance.now(), epoch: 1, repeatRequest: true,
  });
  await vi.advanceTimersByTimeAsync(0);
  event({ type: "session.started" });
  await starting;
  for (const listener of listeners) listener({
    type: "activity", active: false, epoch: 1, startupDrained: true, sent: 100, level: 0,
  });
  await vi.advanceTimersByTimeAsync(2500);
  expect(changed).toHaveBeenLastCalledWith(expect.objectContaining({
    notice: "Dani is listening—please repeat your request.",
  }));
  expect(vi.mocked(post).mock.calls.some(([path, body]) =>
    path.endsWith("/startup") && (body as any)?.state === "ready")).toBe(false);
  userReply("live", "Add a test task.");
  await vi.advanceTimersByTimeAsync(500);
  expect(changed.mock.calls.at(-1)?.[0].notice).toBeUndefined();
  await voice.stop();
});

test.each(["manual","wake"].flatMap(mode=>[0,.0001].map(noise=>({mode,noise}))))(
  "$mode voice completes startup with ongoing capture at noise $noise and stays open past its deadline",async({mode,noise})=>{
    const listeners=new Set<(value:any)=>void>(),changed=vi.fn(),status=vi.fn();
    let latest:any=null,generation=0,wake:(request:string)=>void=()=>{},voice:Voice|null=null;
    const processor=new CaptureProcessor(1000,value=>{
      if(value.type==="activity")latest=value;
      for(const listener of listeners)listener(value);
    });
    const track={enabled:true,stop:vi.fn()};
    const capture={
      stream:{getTracks:()=>[track],getAudioTracks:()=>[track]},microphone:null,context:null,
      get latest(){return latest;},open:vi.fn(async()=>{}),close:vi.fn(),
      subscribe:(listener:(value:any)=>void)=>{listeners.add(listener);return()=>listeners.delete(listener);},
      command:(type:string,epoch?:number)=>processor.command({type,generation,epoch}),
      reset:()=>{generation++;latest=null;processor.command({type:"reset",generation});},
      confirm:async()=>{processor.command({type:"confirm",generation});return "buffered";},
      ready:()=>processor.command({type:"ready",generation}),
      pause:()=>processor.command({type:"pause",generation}),
    };
    const coordinator=new VoiceCoordinator("account","device",{
      capture:()=>capture as never,
      detector:(activate,options)=>{wake=activate;return{start:()=>options.started(),stop:()=>{}};},
      start:async(prepared,attempt)=>{
        voice=new Voice(changed,()=>{},()=>{},draining=>coordinator.draining(draining));
        await voice.start("conversation",undefined,{provider:"live",voice:"marin"},prepared,attempt);
      },
      cancel:async()=>{await voice?.stop();},status,visible:()=>true,
      storage:{getItem:()=>null,setItem:()=>{}},
    });
    const heard:number[]=[];
    const process=async(value:number)=>{
      const output=new Float32Array(100);
      processor.process(new Float32Array(100).fill(value),output);
      heard.push(...output);await vi.advanceTimersByTimeAsync(100);
    };
    let manual:Promise<void>|undefined;
    if(mode==="wake"){
      await coordinator.enable(true,true);
      for(let i=0;i<16;i++)await process(.3);
      wake("");
    }else{
      manual=coordinator.manual();await vi.advanceTimersByTimeAsync(0);
      await process(.3);
    }
    // Keep capturing during provider setup, then replay while capture continues.
    for(let i=0;i<20;i++)await process(noise);
    expect(heard.every(sample=>sample===0)).toBe(true);
    event({type:"session.started"});await vi.advanceTimersByTimeAsync(0);await manual;
    if(mode==="wake")event({type:"session.input_transcript.delta",event_id:"wake",delta:"Hey Dani",start_ms:0,end_ms:1600});
    for(let i=0;i<350;i++)await process(noise);
    expect(coordinator.state).toBe("active_voice");
    expect(capture.close).not.toHaveBeenCalled();
    expect(changed.mock.calls.some(([state])=>state.closed)).toBe(false);
    expect(changed.mock.calls.some(([state])=>typeof state.idle_seconds==="number" && state.idle_seconds<120)).toBe(true);
    expect(heard.filter(sample=>sample>.012)).toEqual(Array((mode==="wake"?16:1)*100).fill(Math.fround(.3)));
    if(mode==="wake")expect(vi.mocked(post).mock.calls.filter(([path,body])=>path.endsWith("/startup") && (body as any)?.state==="ready")).toHaveLength(1);
    await coordinator.dispose();
  },
);
