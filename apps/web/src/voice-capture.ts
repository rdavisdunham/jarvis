import { voiceCaptureFailure, voiceCount, type VoiceCaptureFailure } from "./voice-timing";
import { captureError } from "./monitoring";
import workletUrl from "./voice-worklet.ts?worker&url";
export type CaptureActivity = {type:string; generation:number; active:boolean; epoch:number; captured:number; delivered:number; sent:number; backlog:number; startupDrained:boolean; level:number};
export type CaptureConfirmation = "buffered"|"fresh";
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
  private confirmation: Promise<CaptureConfirmation> | null = null;
  private settleConfirmation: ((result: CaptureConfirmation) => void) | null = null;
  private rejectConfirmation: ((error: Error) => void) | null = null;
  private confirmationTimer: ReturnType<typeof setTimeout> | null = null;
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
      this.node.onprocessorerror = () => {
        if (this.disposed) return;
        this.reportFailure("processor");
        this.cancelConfirmation("Microphone audio processing stopped. Please retry your request.");
        this.failed("Microphone audio processing stopped. Please retry your request.");
      };
      this.node.port.onmessage = event => {
        if (this.disposed || event.data.generation !== this.generation) return;
        if (event.data.type === "confirmed") {
          if (!this.settleConfirmation) return;
          if (event.data.mode === "fresh") this.reportFailure("continuity", event.data.retainedMs);
          this.settleConfirmation(event.data.mode);
          this.clearConfirmation();
          return;
        }
        if (event.data.type === "failure") {
          if(event.data.reason==="overflow")voiceCount("overflow");
          this.reportFailure("overflow");
          this.cancelConfirmation("Voice audio buffer filled or capture failed. Please retry your request.");
          this.failed("Voice audio buffer filled or capture failed. Please retry your request.");
          return;
        }
        if(event.data.type==="activity")this.latest = event.data;
        for (const listener of this.listeners) listener(event.data);
      };
      context.onstatechange = () => { if (!this.disposed && context.state !== "running") this.failed("Microphone audio paused. Tap to resume voice."); };
      for (const track of microphone.getTracks()) track.onended = () => { if (!this.disposed) this.failed("Microphone disconnected. Tap to retry voice."); };
    } catch (error) { this.close(); throw error; }
  }
  command(type: string, epoch?:number) { this.node?.port.postMessage({type,generation:this.generation,epoch}); }
  reset() {
    this.cancelConfirmation();
    this.generation++;
    this.latest = null;
    this.command("reset");
  }
  confirm(): Promise<CaptureConfirmation> {
    if (this.confirmation) return this.confirmation;
    if (this.disposed || !this.node) {
      return Promise.reject(new Error("Microphone capture is not ready. Please retry."));
    }
    this.confirmation = new Promise((resolve, reject) => {
      this.settleConfirmation = resolve;
      this.rejectConfirmation = reject;
      this.confirmationTimer = setTimeout(() => {
        this.reportFailure("confirmation_timeout");
        this.cancelConfirmation("Microphone audio confirmation timed out. Please retry voice.");
      }, 5000);
      this.command("confirm");
    });
    return this.confirmation;
  }
  ready() { this.command("ready"); }
  pause() {
    this.cancelConfirmation();
    this.command("pause");
  }
  private clearConfirmation() {
    if (this.confirmationTimer) clearTimeout(this.confirmationTimer);
    this.confirmationTimer = null;
    this.settleConfirmation = null;
    this.rejectConfirmation = null;
  }
  private cancelConfirmation(message = "Voice audio confirmation cancelled.") {
    this.rejectConfirmation?.(new Error(message));
    this.clearConfirmation();
    this.confirmation = null;
  }
  private reportFailure(reason: VoiceCaptureFailure, retainedMs?: number) {
    voiceCaptureFailure(reason, this.generation, retainedMs);
    captureError(new Error("Voice capture " + reason), { boundary: "voice_capture" });
  }
  close() {
    if (this.disposed) return;
    this.cancelConfirmation();
    this.disposed = true; this.generation++;
    this.node?.disconnect(); this.node?.port.close(); this.node = null;
    this.source?.disconnect(); this.source = null;
    for (const stream of [this.microphone,this.stream]) stream?.getTracks().forEach(t=>t.stop());
    this.stream = this.microphone = null; this.listeners.clear(); this.latest = null;
    if (this.context) { this.context.onstatechange = null; void this.context.close().catch(()=>{}); }
    this.context = null;
  }
}
