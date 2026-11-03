import { isPartialWakePhrase, wakeRequest } from "./assistant-identity";
// Startup-only control; normal conversational playout never enters this gate.
export class StartupCue {
  private ready = false;
  private text = "";
  private active = false;
  private epoch:number;
  private quietAt = performance.now();
  private backlog = true;
  private requestedAt:number|null = null;
  private audible = false;
  private cancelled = false;
  private fallback = false;
  private discarding = false;
  private lastOutput = -Infinity;
  private wakeDelivered = false;
  private sampled = false;
  constructor(private wakeOnly:boolean, epoch:number, private controls:{
    send:(state:"ready"|"cancel",epoch:number)=>void;
    chime:()=>void; interrupt:()=>void; gate:(muted:boolean)=>void;
  }) { this.epoch=epoch; }
  sessionReady(){this.ready=true;}
  get handled(){return this.requestedAt!==null;}
  get eligible(){return this.wakeOnly && !this.cancelled;}
  input(text:string){
    this.text+=text;
    const request=wakeRequest(this.text);
    if(request!==null ? request!=="" : !isPartialWakePhrase(this.text))this.cancel();
  }
  activity(active:boolean, epoch:number, backlog:boolean, delivered:boolean, now:number) {
    if(!this.sampled || active || this.active!==active)this.quietAt=now;
    this.sampled=true;
    this.active=active;this.backlog=backlog;this.wakeDelivered=delivered;
    if(epoch>this.epoch){this.epoch=epoch;this.cancel(now);}
  }
  output(active:boolean, now:number, audible=active) {
    if(active){this.lastOutput=now;if(this.requestedAt!==null && !this.cancelled && audible)this.audible=true;}
  }
  tick(now:number) {
    const quiet=this.sampled && !this.active && !this.backlog && now-this.quietAt>=750;
    if(this.discarding && now-this.lastOutput>=750){this.discarding=false;this.controls.gate(false);}
    if(this.cancelled || !this.wakeOnly)return;
    if(this.audible && quiet && now-this.lastOutput>=750){this.cancelled=true;this.controls.gate(false);return;}
    // Positive recognition of the delivered wake-only utterance is required.
    if(this.ready && this.wakeDelivered && quiet && wakeRequest(this.text)==="" && this.text.trim() && this.requestedAt===null){
      this.requestedAt=now;this.controls.send("ready",this.epoch);
    }
    if(this.requestedAt!==null && !this.audible && !this.fallback && quiet && now-this.requestedAt>=2000){
      this.fallback=true;this.cancel(now);this.controls.chime();
    }
  }
  cancel(now=performance.now()){
    if(this.cancelled)return;this.cancelled=true;
    if(this.requestedAt!==null){
      this.lastOutput=Math.max(this.lastOutput,now);
      this.discarding=true;this.controls.gate(true);this.controls.send("cancel",this.epoch);this.controls.interrupt();
    }
  }
  dispose(){this.cancelled=true;this.discarding=false;this.controls.gate(false);}
}
