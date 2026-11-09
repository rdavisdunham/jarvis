import { VoiceCapture } from "./voice-capture";
import { WakeWord, recognitionType } from "./wake-word";
import { VoiceTiming, voiceCount } from "./voice-timing";
export type ForegroundVoiceState = "off"|"paused_hidden"|"needs_gesture"|"permission_blocked"|"listening_for_wake"|"connecting_voice"|"draining_startup_audio"|"active_voice"|"recovering";
export type VoiceAttempt = {id:string; generation:number; wake:boolean; wakeOnly:boolean; detectedAt:number; epoch:number; repeatRequest?:boolean};
type Capture = Pick<VoiceCapture,"open"|"close"|"reset"|"confirm"|"pause"|"command"|"subscribe"|"latest"|"stream"|"microphone">;
type Dependencies = {
  start:(capture:VoiceCapture, attempt:VoiceAttempt)=>Promise<void>;
  cancel:()=>Promise<void>;
  status:(state:ForegroundVoiceState, message:string, desired:boolean)=>void;
  capture?:(failed:(message:string)=>void)=>Capture;
  detector?:(wake:(request:string)=>void, options:any)=>Pick<WakeWord,"start"|"stop">;
  visible?:()=>boolean;
  storage?:Pick<Storage,"getItem"|"setItem">;
};
export class VoiceCoordinator {
  desired = false;
  preferenceSaved=true;
  state:ForegroundVoiceState = "off";
  private generation = 0;
  private capture:Capture|null = null;
  private detector:Pick<WakeWord,"start"|"stop">|null = null;
  private opening:Promise<void>|null = null;
  private disposed = false;
  private timer:ReturnType<typeof setTimeout>|null = null;
  private deadline:ReturnType<typeof setTimeout>|null = null;
  private retries = 0;
  private unsubscribe:(()=>void)|null = null;
  private activeAttempt:VoiceAttempt|null = null;
  private manualPending=false;
  private recognitionStarted = false;
  private flowing = false;
  private key:string;
  private permissionBlocked=false;
  private listeningSince = 0;
  private listenGeneration = 0;
  constructor(account:string, device:string, private deps:Dependencies) {
    this.key = "eri-wake-v1:"+encodeURIComponent(account)+":"+encodeURIComponent(device);
    try { this.desired = (deps.storage ?? localStorage).getItem(this.key) === "on"; } catch { this.preferenceSaved=false; }
  }
  private visible() { return this.deps.visible?.() ?? document.visibilityState === "visible"; }
  private publish(state:ForegroundVoiceState, message:string) { if(this.disposed)return; this.state=state; this.deps.status(state,message+(this.desired && !this.preferenceSaved?" Preference is saved for this visit only; browser storage is unavailable.":""),this.desired); }
  async enable(enabled:boolean, gesture=false) {
    this.desired=enabled;
    try { (this.deps.storage ?? localStorage).setItem(this.key,enabled?"on":"off");this.preferenceSaved=true; } catch { this.preferenceSaved=false; }
    if (!enabled) {
      this.detector?.stop(); this.detector=null;
      if (this.state === "active_voice") { this.publish("active_voice","Voice active. Wake is off."); return; }
      this.invalidate(); this.capture?.close(); this.capture=null;
      await this.deps.cancel(); this.publish("off","Wake is off. Tap voice to talk."); return;
    }
    await this.resume(gesture);
  }
  async resume(gesture=false) {
    if (this.disposed || !this.desired || this.activeAttempt || this.manualPending) return;
    if(gesture)this.permissionBlocked=false;
    if(this.permissionBlocked){this.publish("permission_blocked","Microphone permission blocked. Allow it in browser settings, then tap Retry wake. Tap voice to talk.");return;}
    if(!this.deps.detector && !recognitionType()){this.publish("needs_gesture","Wake recognition is unavailable in this browser. Tap voice to talk.");return;}
    if (!this.visible()) { this.suspend(); return; }
    if (this.opening) return this.opening;
    const generation = this.generation;
    const run = async () => {
      this.publish("recovering","Preparing wake microphone…");
      try {
        if (!gesture && !this.capture && typeof navigator !== "undefined" && navigator.permissions) {
          const permission = await navigator.permissions.query({name:"microphone" as PermissionName}).catch(()=>null);
          if (generation !== this.generation || this.disposed) return;
          if (permission?.state === "denied") { this.permissionBlocked=true;this.publish("permission_blocked","Microphone permission blocked. Allow it in browser settings, then tap Retry wake. Tap voice to talk."); return; }
          if (permission?.state !== "granted") { this.publish("needs_gesture","Tap Retry wake to enable the microphone and audio on this device."); return; }
        }
        if (!this.capture) {
          let capture:Capture;
          const failed=(message:string)=>{if(this.capture===capture && !this.disposed)this.fail(message);};
          capture = this.deps.capture?.(failed) ?? new VoiceCapture(failed);
          this.capture=capture;
          await capture.open();
          if (generation !== this.generation || this.disposed) { capture.close(); return; }
        }
        this.listen(generation);
      } catch(error) {
        if (generation !== this.generation || this.disposed) return;
        this.capture?.close();this.capture=null;
        this.permissionBlocked=(error as Error).name==="NotAllowedError";
        this.publish((error as Error).name === "NotAllowedError"?"permission_blocked":"needs_gesture",(error as Error).message+" Tap Retry wake or voice to talk.");
      }
    };
    const opening=run();this.opening=opening;
    try { await opening; } finally { if(this.opening===opening)this.opening=null; }
  }
  private listen(generation:number) {
    if (!this.capture || this.disposed || generation!==this.generation || !this.desired || !this.visible()) return;
    this.detector?.stop(); this.unsubscribe?.();
    const listenGeneration = ++this.listenGeneration;
    const valid=()=>generation===this.generation && listenGeneration===this.listenGeneration && !this.disposed;
    this.recognitionStarted=false; this.flowing=false;
    const ready=()=> {
      if (valid() && !this.activeAttempt && this.recognitionStarted && this.flowing) {
        if (this.state!=="listening_for_wake") this.listeningSince=performance.now();
        this.publish("listening_for_wake","Listening for Dani or Hey Dani. Microphone in use by wake and the browser speech service.");
      }
    };
    this.unsubscribe=this.capture.subscribe(activity=> {
      if (!valid()) return;
      this.flowing=activity.captured>0;ready();
    });
    const options = {
      track:this.capture.microphone?.getAudioTracks()[0],
      started:()=> { if(!valid())return; this.recognitionStarted=true;ready(); },
      negative:()=> { if(valid()) this.capture?.command("release"); },
      failed:(reason:string)=> { if(valid()) this.recognitionFailure(reason); },
      ended:()=> { if(valid()) this.retry(false); },
    };
    this.detector=this.deps.detector?.(request=>this.activate(request,generation),options)
      ?? new WakeWord(request=>this.activate(request,generation),()=>{},options);
    this.detector.start();
    if(this.detector && valid()) this.schedule(()=> { if(valid() && (!this.recognitionStarted || !this.flowing))this.fail("Wake capture and recognition did not start together. Please retry."); },5000);

  }
  private recognitionFailure(reason:string) {
    if (this.state==="listening_for_wake" && performance.now()-this.listeningSince>=10000) this.retries=0;
    if (["not-allowed","service-not-allowed","audio-capture"].includes(reason)) {
      this.permissionBlocked=true;this.listenGeneration++;this.clearTimers();
      this.detector?.stop();this.capture?.close();this.capture=null;
      this.publish("permission_blocked","Wake paused: microphone permission or capture unavailable. Tap Retry wake or voice to talk.");
    } else this.retry(true);
  }
  private retry(failure:boolean) {
    voiceCount("recovery");
    this.listenGeneration++;this.detector?.stop();this.detector=null;
    if (!this.desired || this.activeAttempt || !this.visible() || this.disposed) return;
    if (failure && this.retries>=4) { this.capture?.close();this.capture=null;this.publish("needs_gesture","Wake paused after repeated recognition failures. Tap Retry wake or voice to talk.");return; }
    const delay=failure?[500,1000,2000,4000][this.retries++]:500;
    this.publish("recovering","Wake recognition recovering…");
    const generation=this.generation;
    this.schedule(()=>this.listen(generation),delay);
  }
  private schedule(action:()=>void, delay:number) { if(this.timer)clearTimeout(this.timer);this.timer=setTimeout(action,delay); }
  private async activate(request:string,generation:number) {
    if (generation!==this.generation || this.activeAttempt || !this.capture || !this.desired || !this.visible() || this.disposed) {voiceCount("deduplication");return;}
    this.listenGeneration++;this.detector?.stop();this.detector=null;this.unsubscribe?.();this.unsubscribe=null;
    this.clearTimers();
    const capture=this.capture;
    const attempt:VoiceAttempt={id:crypto.randomUUID(),generation,wake:true,wakeOnly:request==="",detectedAt:performance.now(),epoch:capture.latest?.epoch??0};
    this.activeAttempt=attempt;
    voiceCount("detection");new VoiceTiming(attempt.id,attempt.detectedAt).mark("detected");
    this.publish("connecting_voice","Connecting voice; keeping your first request…");
    this.deadline=setTimeout(()=> { if(this.activeAttempt===attempt && this.state!=="active_voice")this.fail("Voice startup timed out. Please retry your request; check Activity for accepted work."); },30000);
    try {
      const confirmation=await capture.confirm();
      if(this.activeAttempt!==attempt)return;
      new VoiceTiming(attempt.id,attempt.detectedAt).mark("confirmation");
      if(confirmation==="fresh") {
        attempt.repeatRequest=true;
        attempt.wakeOnly=false;
        this.publish("connecting_voice","Connecting Dani; please repeat your request when voice is ready.");
      }
      await this.deps.start(capture as VoiceCapture,attempt);
    } catch(error) { if(this.activeAttempt===attempt)this.fail((error as Error).message); }
  }
  draining(active:boolean) {
    if(!this.activeAttempt)return;
    this.publish(active?"draining_startup_audio":"active_voice",active?"Finishing startup audio…":this.activeAttempt.repeatRequest?"Dani is listening—please repeat your request.":"Voice active.");
    if(!active && this.deadline) { clearTimeout(this.deadline);this.deadline=null; }
  }
  async manual() {
    if(this.disposed || this.activeAttempt || this.manualPending)return;
    this.manualPending=true;
    const generation=++this.generation;
    this.clearTimers();this.listenGeneration++;this.detector?.stop();this.detector=null;
    this.unsubscribe?.();this.unsubscribe=null;
    if(this.opening){this.capture?.close();this.capture=null;}
    this.publish("connecting_voice","Connecting voice…");
    try {
      if(!this.capture){
        let capture:Capture;
        const failed=(message:string)=>{if(this.capture===capture && !this.disposed)this.fail(message);};
        capture=this.deps.capture?.(failed) ?? new VoiceCapture(failed);this.capture=capture;
        await capture.open();
        if(this.disposed || generation!==this.generation){capture.close();return;}
      }else this.capture.reset();
      const capture=this.capture;
      const attempt:VoiceAttempt={id:crypto.randomUUID(),generation,wake:false,wakeOnly:false,detectedAt:performance.now(),epoch:this.capture.latest?.epoch??0};
      this.activeAttempt=attempt;
      this.deadline=setTimeout(()=>{if(this.activeAttempt===attempt && this.state!=="active_voice")this.fail("Voice startup timed out. Please retry; check Activity for accepted work.");},30000);
      await capture.confirm();
      if(this.activeAttempt!==attempt)return;
      new VoiceTiming(attempt.id,attempt.detectedAt).mark("confirmation");
      await this.deps.start(capture as VoiceCapture,attempt);
    }catch(error){if(generation===this.generation)this.fail((error as Error).message);}
    finally{if(generation===this.generation)this.manualPending=false;}
  }
  async cancelStartup() {
    this.invalidate();this.detector?.stop();this.detector=null;
    if(this.capture?.stream && this.desired)this.capture.reset();
    else {this.capture?.close();this.capture=null;}
    await this.deps.cancel();await this.ended();
  }
  localEnd() {
    if(this.disposed)return;
    this.capture?.pause();
    if(!this.desired){this.capture?.close();this.capture=null;}
    this.publish("recovering","Voice closing; saved work continues.");
  }
  async suspendContext() {
    this.invalidate();this.detector?.stop();this.detector=null;this.capture?.close();this.capture=null;
    this.publish("recovering","Wake paused while the account context changes.");
    await this.deps.cancel();
  }
  async ended() {
    const generation=++this.generation;
    this.manualPending=false;this.opening=null;
    this.clearTimers(); this.activeAttempt=null; this.capture?.reset();
    this.unsubscribe?.();this.unsubscribe=null;
    if (!this.desired) { this.capture?.close();this.capture=null;this.publish("off","Voice ended. Wake is off.");return; }
    if(!this.visible()){this.suspend();return;}
    if(generation===this.generation)await this.resume();
  }
  suspend() {
    if (this.state==="active_voice")return;
    const hadAttempt=!!this.activeAttempt;
    this.invalidate();this.detector?.stop();this.detector=null;
    this.capture?.close();this.capture=null;
    if(hadAttempt)void this.deps.cancel();
    this.publish(this.desired?"paused_hidden":"off",this.desired?"Wake paused while this page is hidden.":"Wake is off.");
  }
  private fail(message:string) {
    this.invalidate();this.detector?.stop();this.detector=null;this.capture?.close();this.capture=null;
    void this.deps.cancel();this.publish("needs_gesture",message+" Tap Retry wake or voice to talk.");
  }
  private clearTimers(){if(this.timer)clearTimeout(this.timer);if(this.deadline)clearTimeout(this.deadline);this.timer=this.deadline=null;}
  private invalidate(){this.opening=null;this.manualPending=false;this.generation++;this.activeAttempt=null;this.clearTimers();this.unsubscribe?.();this.unsubscribe=null;}
  dispose() { this.invalidate();this.detector?.stop();this.detector=null;this.capture?.close();this.capture=null;this.disposed=true;return this.deps.cancel(); }
}
