import { startRecognition, validatesAudioTrack } from "./voice-recognition";
import { WAKE_HINTS, wakeRequest } from "./assistant-identity";
export { wakeRequest } from "./assistant-identity";
// Foreground browser speech service. Detection hints never execute a task.
export const isWakePhrase = (text: string) => wakeRequest(text) === "";
export type Recognition = {
  continuous:boolean; interimResults:boolean; maxAlternatives:number; lang:string;
  processLocally?:boolean; phrases?:unknown[];
  onstart:(()=>void)|null; onresult:((event:any)=>void)|null;
  onerror:((event:any)=>void)|null; onend:(()=>void)|null;
  start(track?:MediaStreamTrack):void; abort():void;
};
export function recognitionType(): (new () => Recognition) | undefined {
  if (typeof window === "undefined") return undefined;
  const browser = window as any;
  return browser.SpeechRecognition ?? browser.webkitSpeechRecognition;
}
export class WakeWord {
  private recognition:Recognition|null = null;
  private enabled = false;
  private stable:ReturnType<typeof setTimeout>|null = null;
  private candidate = "";
  private request = "";
  private bias = true;
  constructor(private wake:(request:string)=>void, private status:(message:string)=>void,
    private options:{started?:()=>void; failed?:(reason:string)=>void; ended?:()=>void; negative?:()=>void;
      track?:MediaStreamTrack; sharedTrackValidated?:boolean} = {}) {}
  start() {
    this.stop();
    const Type = recognitionType();
    if (!Type) { this.options.failed?.("unavailable"); this.status("Wake unavailable. Tap voice to talk."); return; }
    this.enabled = true;
    const sharedTrackValidated = this.options.sharedTrackValidated ?? validatesAudioTrack(Type,this.options.track);
    const recognition = new Type(); this.recognition = recognition;
    recognition.continuous = true; recognition.interimResults = true; recognition.maxAlternatives = 3; recognition.lang = "en-US";
    if ("processLocally" in recognition) recognition.processLocally = false;
    // Bias is optional and must not prevent the existing browser-service path.
    const Phrase = (window as any).SpeechRecognitionPhrase;
    if (this.bias && "phrases" in recognition && Phrase) {
      try { recognition.phrases = WAKE_HINTS.map(phrase => new Phrase(phrase,5)); } catch { /* optional */ }
    }
    recognition.onstart = () => { if (this.recognition === recognition && this.enabled) this.options.started?.(); };
    recognition.onresult = event => {
      for (let i=event.resultIndex; i<event.results.length; i++) {
        if (!this.enabled || this.recognition !== recognition) return;
        const result = event.results[i];
        let request:string|null = null, prefix = "";
        for (let a=0; a<Math.min(result.length ?? 1,3); a++) {
          const text = result[a].transcript;
          request = wakeRequest(text);
          if (request !== null) { prefix = text.trim().slice(0,text.trim().length-request.length).toLowerCase().replace(/[^a-z]/g,""); break; }
        }
        if (request === null) {
          this.cancelCandidate();
          if (result.isFinal) this.options.negative?.();
          continue;
        }
        this.request = request;
        // Stability belongs to the wake prefix, not the still-growing request suffix.
        const key = String(i)+":"+prefix;
        if (result.isFinal) { this.activate(); return; }
        if (key !== this.candidate) {
          this.cancelCandidate(); this.candidate = key;
          this.stable = setTimeout(()=> { if (this.enabled && this.recognition === recognition) this.activate(); },200);
        }
      }
    };
    recognition.onerror = event => {
      if (!this.enabled || this.recognition !== recognition) return;
      this.cancelCandidate();
      if (event.error === "phrases-not-supported" && this.bias) { this.bias=false; this.start(); return; }
      if (event.error === "audio-capture" && sharedTrackValidated) { this.options.sharedTrackValidated=false; this.start(); return; }
      if (["no-speech","aborted"].includes(event.error)) { this.options.negative?.(); return; }
      this.stop(); this.options.failed?.(event.error);
    };
    recognition.onend = () => {
      if (!this.enabled || this.recognition !== recognition) return;
      this.stop(); this.options.ended?.();
    };
    try { startRecognition(recognition,this.options.track,sharedTrackValidated); }
    catch (e) {
      if(sharedTrackValidated && (e as Error).name!=="NotAllowedError") { this.options.sharedTrackValidated=false; this.start();return; }
      this.stop(); this.options.failed?.((e as Error).name === "NotAllowedError" ? "not-allowed" : "start_failed"); }
  }
  private activate() { const request = this.request; this.stop(); this.wake(request); }
  private cancelCandidate() { if (this.stable) clearTimeout(this.stable); this.stable = null; this.candidate = ""; }
  stop() {
    this.enabled = false; this.cancelCandidate();
    if (this.recognition) {
      const r = this.recognition; this.recognition = null;
      r.onstart = r.onresult = r.onerror = r.onend = null;
      try { r.abort(); } catch { /* already ended */ }
    }
  }
}
