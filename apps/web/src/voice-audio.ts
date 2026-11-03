// One monotonic sample timeline, shared by the worklet and the offline harness.
// No samples leave this object until confirm + ready. Never log/persist samples.
export type AudioBlock = {start:number; samples:Float32Array; silent:boolean};
export class AudioQueue {
  private blocks: AudioBlock[] = [];
  written = 0;
  sent = 0;
  cursor = 0;
  pinned: number | null = null;
  confirmed = false;
  ready = false;
  failed = false;
  constructor(readonly rate: number, readonly maxBytes = 16 * 1024 * 1024) {}
  get retained() { return this.written - (this.blocks[0]?.start ?? this.written); }
  get backlog() { return this.confirmed ? Math.max(0, this.written - this.cursor) : 0; }
  push(samples: Float32Array, voiced: boolean) {
    if (this.failed) return;
    const start = this.written;
    this.written += samples.length;
    if (voiced && this.pinned === null && !this.confirmed)
      this.pinned = Math.max(this.blocks[0]?.start ?? start, start - Math.round(this.rate * .2));
    // Only near-digital silence is safe to compact; quiet or uncertain speech is retained.
    this.blocks.push({start, samples:samples.slice(), silent:samples.every(x => Math.abs(x) < 0.00001)});
    this.trim(this.confirmed ? this.cursor : this.pinned ?? this.written - 3 * this.rate);
    if (this.retained > 60 * this.rate || this.retained * 4 > this.maxBytes) {
      this.clear(); this.failed = true;
      throw new Error("Voice audio buffer filled. Please retry your request.");
    }
  }
  release() {
    if (this.confirmed) return;
    this.pinned = null; this.trim(this.written - 3 * this.rate);
  }
  confirm(requiredStart?:number) {
    if (this.failed) throw new Error("Voice audio capture failed. Please retry.");
    if (this.confirmed) return;
    const oldest=this.blocks[0]?.start ?? this.written;
    if(requiredStart!==undefined && (oldest>requiredStart || (this.pinned!==null && this.pinned!==requiredStart))){
      this.clear();this.failed=true;throw new Error("Wake audio continuity is uncertain. Please repeat your request.");
    }
    this.confirmed = true;
    this.cursor = Math.max(oldest, this.pinned ?? requiredStart ?? this.written - 3 * this.rate);
    this.trim(this.cursor);
  }
  read(count: number) {
    const output = new Float32Array(count);
    if (!this.ready || !this.confirmed || this.failed) return output;
    let offset = 0;
    while (offset < count && this.cursor < this.written) {
      this.trim(this.cursor);
      let block = this.blocks[0];
      if (!block) break;
      // Remove surplus known silence only, preserving >=200 ms at every boundary.
      if (block.silent && this.backlog > this.rate * .4) {
        let silentEnd = this.cursor;
        for (const b of this.blocks) { if (!b.silent) break; silentEnd = b.start + b.samples.length; }
        const skip = Math.max(0, silentEnd - this.cursor - Math.ceil(this.rate * .2));
        this.cursor += skip; this.trim(this.cursor); block = this.blocks[0];
        if (!block) break;
      }
      const from = this.cursor - block.start;
      const n = Math.min(count - offset, block.samples.length - from);
      output.set(block.samples.subarray(from, from + n), offset);
      offset += n; this.cursor += n;this.sent+=n;
    }
    this.trim(this.cursor);
    return output;
  }
  private trim(before: number) {
    while (this.blocks.length && this.blocks[0].start + this.blocks[0].samples.length <= before) this.blocks.shift();
  }
  clear() { this.blocks = []; this.pinned = null; this.confirmed = this.ready = false; this.cursor = this.written;this.sent=0; }
}

// Processor core is independent of AudioWorklet globals so tests execute the real path.
export class CaptureProcessor {
  queue: AudioQueue;
  generation = 0;
  active = false;
  lastVoice = -Infinity;
  epoch = 0;
  ticks = 0;
  private utteranceStart:number|null=null;
  private awaitOnset=false;
  private cueArmed=false;
  private cueEpoch=0;
  constructor(readonly rate: number, private publish: (message: Record<string, any>) => void) {
    this.queue = new AudioQueue(rate);
  }
  command(message: {generation:number; type:string; epoch?:number}) {
    if (message.type === "reset") {
      if (message.generation < this.generation) return;
      this.generation = message.generation;
      this.queue.clear(); this.queue.failed = false; this.active = false;this.lastVoice=-Infinity;this.cueArmed=false;this.utteranceStart=null;this.awaitOnset=false;
      return;
    }
    if (message.generation !== this.generation) return;
    if(message.type==="cue_arm") {
      if(this.active || message.epoch!==this.epoch){this.status("cue_cancel");return;}
      this.cueArmed=true;this.cueEpoch=this.epoch;this.status("cue_armed");
    }
    if(message.type==="cue_disarm")this.cueArmed=false;
    if (message.type === "confirm") {
      try {this.queue.confirm(this.utteranceStart===null?undefined:Math.max(0,this.utteranceStart-Math.round(this.rate*.2)));}
      catch {this.publish({type:"failure",generation:this.generation,reason:"continuity"});}
    }
    if (message.type === "ready") this.queue.ready = true;
    if (message.type === "release" && !this.queue.confirmed) {this.queue.release();this.awaitOnset=true;}
    if (message.type === "pause") this.queue.clear();
  }
  process(input: Float32Array, output: Float32Array) {
    const rms = Math.sqrt(input.reduce((sum,x) => sum + x*x, 0) / (input.length || 1));
    const voiced = rms > .012;
    if (voiced) this.lastVoice = this.queue.written;
    const active = voiced || this.queue.written - this.lastVoice < .2 * this.rate;
    if (active && !this.active) {
      this.epoch++;
      if(!this.queue.confirmed){this.utteranceStart=this.queue.written;this.awaitOnset=false;}
    }
    this.active = active;
    if(this.cueArmed && this.epoch>this.cueEpoch){
      // Hold new speech before its first output frame until stale greeting audio
      // drains. Input remains captured in this same ordered queue.
      this.cueArmed=false;this.queue.ready=false;this.status("cue_cancel");
    }
    try {
      this.queue.push(input, active && !this.awaitOnset);
      output.set(this.queue.read(output.length));
      if (++this.ticks % 16 === 0) this.status("activity",Math.min(1,rms*7));
    } catch {
      output.fill(0);
      this.publish({type:"failure", generation:this.generation,reason:"overflow"});
    }
  }
  private status(type:string,level=0) {
    this.publish({type,generation:this.generation,active:this.active,epoch:this.epoch,
      captured:this.queue.written,delivered:this.queue.cursor,sent:this.queue.sent,backlog:this.queue.backlog,level});
  }

}
