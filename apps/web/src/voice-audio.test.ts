import { expect, test } from "vitest";
import { AudioQueue, CaptureProcessor } from "./voice-audio";
test.each([0,2,5,15].flatMap(delay => [0,.0001,.1].map(noise => ({delay,noise}))))(
  "startup completes with continuous capture after $delay seconds setup at noise $noise", ({delay,noise}) => {
    const messages:Record<string,any>[]=[];
    const processor=new CaptureProcessor(1000,message=>messages.push(message));
    const output=new Float32Array(100),captured:number[]=[],heard:number[]=[];
    const process=(value:number,delivering=false)=>{
      const input=new Float32Array(100).fill(value);
      captured.push(...input);processor.process(input,output);
      if(delivering)heard.push(...output);
      else expect(output.every(sample=>sample===0)).toBe(true);
    };
    process(.3);processor.command({type:"confirm",generation:0});
    for(let i=0;i<delay*10;i++)process(noise);
    processor.command({type:"ready",generation:0});
    for(let i=0;i<delay*10+32;i++)process(noise,true);
    expect(messages.filter(message=>message.type==="activity").at(-1)?.startupDrained).toBe(true);
    // A replenished queue may retain silence or normal transport delay after startup.
    expect(processor.queue.backlog).toBeGreaterThan(0);
    expect(heard.filter(sample=>sample!==0)).toEqual(captured.slice(0,heard.length).filter(sample=>sample!==0));
  },
);

test("repeated ready commands keep the initial delivery boundary; reset starts a fresh one",()=>{
  const messages:Record<string,any>[]=[];
  const processor=new CaptureProcessor(1000,message=>messages.push(message));
  const input=new Float32Array(100).fill(.3),output=new Float32Array(100);
  processor.process(input,output);processor.process(input,output);
  processor.command({type:"confirm",generation:0});processor.command({type:"ready",generation:0});
  processor.process(input,output);
  processor.command({type:"ready",generation:0});
  processor.process(input,output);
  expect(processor.queue.startupDrained).toBe(true);
  processor.command({type:"reset",generation:1});
  processor.command({type:"ready",generation:0});
  processor.process(new Float32Array(100),output);processor.process(new Float32Array(100),output);
  for(let i=0;i<16;i++)processor.process(input,output);
  expect(messages.at(-1)?.startupDrained).toBe(false);
  processor.command({type:"confirm",generation:1});processor.command({type:"ready",generation:1});
  processor.process(input,output);
  expect(processor.queue.startupDrained).toBe(false);
  for(let i=0;i<18;i++)processor.process(input,output);
  expect(processor.queue.startupDrained).toBe(true);
  expect(messages.some(message=>message.type==="failure")).toBe(false);
});

test.each([0,2,5,15])("ordered voiced frames through %s seconds startup, including live append", delay => {
  const rate = 1000, q = new AudioQueue(rate);
  const expected:number[] = [];
  // Three discarded pre-wake seconds, then current utterance pinned at onset.
  for (let i=0;i<30;i++) q.push(new Float32Array(100),false);
  for (let i=0;i<Math.max(1,delay*10);i++) {
    const block = Float32Array.from({length:100},(_,j)=>(i*100+j+1)/10000);
    expected.push(...block); q.push(block,true);
  }
  q.confirm(); expect(q.read(100).every(x=>x===0)).toBe(true);
  q.ready = true;
  const heard:number[] = [];
  while (q.backlog) heard.push(...q.read(Math.min(100,q.backlog)));
  const next = new Float32Array([.8,.9,1]); q.push(next,true); heard.push(...q.read(3));
  expect(heard.filter(x=>x!==0)).toEqual([...expected,...next]);
  expect(q.backlog).toBe(0);
});
test("known silence compacts with 200ms retained; quiet speech never compacts", () => {
  const q = new AudioQueue(1000); q.push(new Float32Array([.1]),true);
  q.push(new Float32Array(2000),false); q.push(new Float32Array([.2]),true);
  q.confirm(); q.ready=true;
  const heard = Array.from(q.read(202));
  expect(heard).toEqual([expect.closeTo(.1),...Array(200).fill(0),expect.closeTo(.2)]);
  const quiet = new AudioQueue(1000); quiet.push(new Float32Array(1000).fill(.0001),true);
  quiet.confirm(); quiet.ready=true; expect(quiet.read(1000)).toEqual(new Float32Array(1000).fill(.0001));
});
test("bounded queue fails explicitly instead of delivering partial speech", () => {
  const q = new AudioQueue(1000, 400); q.push(new Float32Array(100).fill(.1),true);
  expect(()=>q.push(new Float32Array([.2]),true)).toThrow(/buffer filled/);
  expect(q.failed).toBe(true); expect(q.read(10)).toEqual(new Float32Array(10));
});
test("real processor core honors generation, silence-before-ready, cancellation and sample continuity", () => {
  const messages:Record<string,any>[]=[];
  const processor = new CaptureProcessor(1000,m=>messages.push(m));
  const frame = new Float32Array(100).fill(.3), out = new Float32Array(100);
  processor.process(frame,out); expect(out.every(x=>x===0)).toBe(true);
  processor.command({type:"confirm",generation:0});
  processor.process(frame,out); expect(out.every(x=>x===0)).toBe(true);
  processor.command({type:"ready",generation:0});
  processor.process(frame,out); expect(out).toEqual(frame);
  processor.command({type:"reset",generation:1});
  processor.command({type:"ready",generation:0});
  processor.process(frame,out); expect(out.every(x=>x===0)).toBe(true);
  expect(processor.queue.ready).toBe(false);
});
test("negative recognition releases a pin while retaining rolling prebuffer", () => {
  const q = new AudioQueue(1000); for(let i=0;i<50;i++) q.push(new Float32Array(100).fill(.1),true);
  expect(q.retained).toBe(5000); q.release(); expect(q.retained).toBe(3000);
});

test("continuous speech never loses a voiced boundary frame while capture keeps appending",()=>{
 const q=new AudioQueue(1000);const heard:number[]=[],expected:number[]=[];
 for(let i=0;i<30;i++){const frame=Float32Array.from({length:100},(_,j)=>(i*100+j+1)/10000);q.push(frame,true);expected.push(...frame);}
 q.confirm();q.ready=true;
 for(let i=30;i<60;i++){const frame=Float32Array.from({length:100},(_,j)=>(i*100+j+1)/10000);q.push(frame,true);expected.push(...frame);heard.push(...q.read(100));}
 while(q.backlog)heard.push(...q.read(Math.min(100,q.backlog)));
 expect(heard).toEqual(expected);expect(q.cursor).toBe(q.written);
});
test("sixty-second audio limit is independent of byte cap and cancellation clears all samples",()=>{
 const q=new AudioQueue(10);q.push(new Float32Array(600).fill(.2),true);
 expect(()=>q.push(new Float32Array([.3]),true)).toThrow();
 expect(q.retained).toBe(0);
 const p=new CaptureProcessor(1000,()=>{});p.process(new Float32Array(100).fill(.3),new Float32Array(100));
 p.command({type:"pause",generation:0});expect(p.queue.retained).toBe(0);expect(p.queue.confirmed).toBe(false);
});

test("bundled worklet adapter uses the tested processor core and drops old-generation commands",async()=>{
 const {vi}=await import("vitest");let Type:any;
 class Base {port={postMessage:vi.fn(),onmessage:null as any};}
 vi.stubGlobal("AudioWorkletProcessor",Base);vi.stubGlobal("sampleRate",1000);
 vi.stubGlobal("registerProcessor",(name:string,processor:any)=>{expect(name).toBe("eri-voice-capture");Type=processor;});
 try{
   await import("./voice-worklet");const worklet=new Type();
   const frame=new Float32Array(100).fill(.4),output=new Float32Array(100);
   worklet.process([[frame]],[[output]]);expect(output.every(x=>x===0)).toBe(true);
   worklet.port.onmessage({data:{type:"confirm",generation:0}});worklet.port.onmessage({data:{type:"ready",generation:0}});
   worklet.process([[frame]],[[output]]);expect(output).toEqual(frame);
   worklet.port.onmessage({data:{type:"reset",generation:1}});worklet.port.onmessage({data:{type:"ready",generation:0}});
   worklet.process([[frame]],[[output]]);expect(output.every(x=>x===0)).toBe(true);
 }finally{vi.unstubAllGlobals();}
});

test("startup cue barge-in holds the very first voiced frame until stale output drains",()=>{
 const messages:Record<string,any>[]=[];const p=new CaptureProcessor(1000,m=>messages.push(m));
 p.command({type:"confirm",generation:0});p.command({type:"ready",generation:0});
 p.process(new Float32Array(100),new Float32Array(100));expect(p.queue.startupDrained).toBe(true);
 p.command({type:"cue_arm",generation:0,epoch:0});expect(messages.at(-1)?.type).toBe("cue_armed");
 const request=Float32Array.from({length:100},(_,i)=>(i+1)/1000),out=new Float32Array(100);
 p.process(request,out);expect(out.every(x=>x===0)).toBe(true);expect(messages.some(x=>x.type==="cue_cancel")).toBe(true);
 p.command({type:"cue_disarm",generation:0});p.command({type:"ready",generation:0});
 expect(p.queue.startupDrained).toBe(true);
 expect(p.queue.read(100)).toEqual(request);
});

test("late negative recognition switches to fresh audio and discards the old utterance's tail",()=>{
 const messages:Record<string,any>[]=[];const p=new CaptureProcessor(1000,m=>messages.push(m));
 for(let i=0;i<50;i++)p.process(new Float32Array(100).fill(.2),new Float32Array(100));
 p.command({type:"release",generation:0});
 p.process(new Float32Array(100).fill(.2),new Float32Array(100));
 p.command({type:"confirm",generation:0});p.command({type:"ready",generation:0});
 expect(messages).toContainEqual(expect.objectContaining({type:"confirmed",mode:"fresh"}));
 expect(messages.some(m=>m.type==="failure")).toBe(false);
 expect(p.queue.read(100)).toEqual(new Float32Array(100));
 const output=new Float32Array(100);
 p.process(new Float32Array(100).fill(.3),output);
 expect(output.every(sample=>sample===0)).toBe(true);
 for(let i=0;i<4;i++)p.process(new Float32Array(100),output);
 const repeated=new Float32Array(100).fill(.4);
 p.process(repeated,output);
 expect(output).toEqual(repeated);
});

test.each([1000,44100,48000])("pauses preserve one candidate boundary at %s Hz with real-sized frames",rate=>{
 const messages:Record<string,any>[]=[];
 const p=new CaptureProcessor(rate,message=>messages.push(message));
 const output=new Float32Array(128),expected:number[]=[];
 const capture=(seconds:number,value:number)=>{
   for(let i=0;i<Math.ceil(seconds*rate/128);i++){
     const input=new Float32Array(128).fill(value);
     p.process(input,output);
     if(value)expected.push(...input);
     expect(output.every(sample=>sample===0)).toBe(true);
   }
 };
 capture(4,0);
 capture(.35,.2);
 const start=p.queue.pinned;
 capture(.35,0);
 capture(.35,.3);
 expect(p.queue.pinned).toBe(start);
 p.command({type:"confirm",generation:0});
 expect(messages).toContainEqual(expect.objectContaining({type:"confirmed",mode:"buffered"}));
 expect(messages.some(message=>message.type==="failure")).toBe(false);
 p.command({type:"ready",generation:0});
 const heard=p.queue.read(p.queue.backlog);
 expect(Array.from(heard).filter(sample=>sample!==0)).toEqual(expected);
});

test("negative recognition excludes earlier audio while a new multi-part wake stays intact",()=>{
 const messages:Record<string,any>[]=[];
 const p=new CaptureProcessor(1000,message=>messages.push(message));
 const output=new Float32Array(100);
 const capture=(count:number,value:number)=>{
   for(let i=0;i<count;i++)p.process(new Float32Array(100).fill(value),output);
 };
 capture(5,.1);capture(4,0);
 p.command({type:"release",generation:0});
 capture(35,0);capture(4,.3);capture(4,0);capture(4,.4);
 p.command({type:"confirm",generation:0});p.command({type:"ready",generation:0});
 expect(messages).toContainEqual(expect.objectContaining({type:"confirmed",mode:"buffered"}));
 expect(Array.from(p.queue.read(p.queue.backlog)).filter(sample=>sample!==0))
   .toEqual([...Array(400).fill(Math.fround(.3)),...Array(400).fill(Math.fround(.4))]);
});

test("confirmation of an overflowing buffer fails and never reports fresh recovery",()=>{
 const messages:Record<string,any>[]=[];
 const p=new CaptureProcessor(10,message=>messages.push(message));
 p.process(new Float32Array(601).fill(.2),new Float32Array(601));
 p.command({type:"confirm",generation:0});
 expect(messages).toContainEqual(expect.objectContaining({type:"failure",reason:"overflow"}));
 expect(messages.some(message=>message.type==="confirmed")).toBe(false);
});
test("negative recognition followed by a new utterance resets the pin at its actual onset",()=>{
 const p=new CaptureProcessor(1000,()=>{});
 p.process(new Float32Array(100).fill(.3),new Float32Array(100));p.command({type:"release",generation:0});
 for(let i=0;i<35;i++)p.process(new Float32Array(100),new Float32Array(100));
 const wake=new Float32Array(100).fill(.4);p.process(wake,new Float32Array(100));p.command({type:"confirm",generation:0});p.command({type:"ready",generation:0});
 expect(p.queue.read(p.queue.backlog).filter(x=>x!==0)).toEqual(wake);
});
