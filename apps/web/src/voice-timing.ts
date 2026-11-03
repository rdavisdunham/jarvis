// Bounded, content-free client diagnostics. No speech, samples, names or prompts.
export type VoiceStage = "detected"|"capture"|"preparation"|"sdp"|"create_request"|"create_response"|"data_channel"|"session_started"|"first_frame"|"drained"|"first_audible";
const traces: {attempt:string; stage:VoiceStage; elapsed_ms:number}[] = [];
const counters:Record<string,number> = {};
export function voiceCount(event:"detection"|"deduplication"|"recovery"|"overflow"|"stale_ending") { counters[event]=(counters[event]??0)+1; }
export class VoiceTiming {
  private seen = new Set<VoiceStage>();
  constructor(readonly attempt:string, private start=performance.now()) {}
  mark(stage:VoiceStage) {
    if(this.seen.has(stage) || traces.some(t=>t.attempt===this.attempt && t.stage===stage))return; this.seen.add(stage);
    traces.push({attempt:this.attempt,stage,elapsed_ms:Math.max(0,performance.now()-this.start)});
    if(traces.length>256)traces.shift();
  }
}
export const voiceDiagnostics=()=>({traces:traces.map(x=>({...x})),counters:{...counters}});
