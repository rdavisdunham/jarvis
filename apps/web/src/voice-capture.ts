import { voiceCount } from "./voice-timing";
import workletUrl from "./voice-worklet.ts?worker&url";
export type CaptureActivity = {type:string; generation:number; active:boolean; epoch:number; captured:number; delivered:number; sent:number; backlog:number; level:number};
export class VoiceCapture {
  private node: AudioWorkletNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private destination: MediaStreamAudioDestinationNode | null = null;
  context: AudioContext | null = null;
  microphone: MediaStream | null = null;
  stream: MediaStream | null = null;
  generation = 0;
  latest: CaptureActivity | null = null;
  private disposed = false;
  private listeners = new Set<(activity:CaptureActivity) => void>();
  constructor(private failed: (message:string) => void) {}
  subscribe(listener: (activity:CaptureActivity) => void) { this.listeners.add(listener); return () => { this.listeners.delete(listener); }; }
  async open() {
    // Construct/resume in the opt-in gesture before awaiting microphone permission.
    const context = new AudioContext(); this.context = context;
    const resumed = context.resume().catch(()=>{});
    try {
      const microphone = await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
      if (this.disposed) { microphone.getTracks().forEach(t=>t.stop()); return; }
      this.microphone = microphone;
      await Promise.all([resumed, context.audioWorklet.addModule(workletUrl)]);
      if (this.disposed) return;
      if (context.state !== "running") throw new Error("Tap to enable microphone audio, then retry wake.");
      this.source = context.createMediaStreamSource(microphone);
      this.node = new AudioWorkletNode(context,"eri-voice-capture",{numberOfInputs:1,numberOfOutputs:1,outputChannelCount:[1],channelCount:1,channelCountMode:"explicit"});
      this.destination = context.createMediaStreamDestination();
      this.source.connect(this.node).connect(this.destination);
      this.stream = this.destination.stream;
      this.node.onprocessorerror=()=>{if(!this.disposed)this.failed("Microphone audio processing stopped. Please retry your request.");};
      this.node.port.onmessage = event => {
        if (this.disposed || event.data.generation !== this.generation) return;
        if (event.data.type === "failure") {
          if(event.data.reason==="overflow")voiceCount("overflow");
          this.failed(event.data.reason==="continuity"?"Wake audio could not be matched safely. Please repeat your request.":"Voice audio buffer filled or capture failed. Please retry your request.");return;
        }
        if(event.data.type==="activity")this.latest = event.data;
        for (const listener of this.listeners) listener(event.data);
      };
      context.onstatechange = () => { if (!this.disposed && context.state !== "running") this.failed("Microphone audio paused. Tap to resume voice."); };
      for (const track of microphone.getTracks()) track.onended = () => { if (!this.disposed) this.failed("Microphone disconnected. Tap to retry voice."); };
    } catch (error) { this.close(); throw error; }
  }
  command(type: string, epoch?:number) { this.node?.port.postMessage({type,generation:this.generation,epoch}); }
  reset() { this.generation++; this.latest = null; this.command("reset"); }
  confirm() { this.command("confirm"); }
  ready() { this.command("ready"); }
  pause() { this.command("pause"); }
  close() {
    if (this.disposed) return;
    this.disposed = true; this.generation++;
    this.node?.disconnect(); this.node?.port.close(); this.node = null;
    this.source?.disconnect(); this.source = null;
    for (const stream of [this.microphone,this.stream]) stream?.getTracks().forEach(t=>t.stop());
    this.stream = this.microphone = null; this.listeners.clear(); this.latest = null;
    if (this.context) { this.context.onstatechange = null; void this.context.close().catch(()=>{}); }
    this.context = null;
  }
}
