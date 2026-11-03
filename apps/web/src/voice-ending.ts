// Conservative command vocabulary; unknown/uncertain intent always keeps listening.
export type ClosingQuestion = "more" | "done";
export function normalize(text: string) {
  return text.toLowerCase().replace(/[’']/g, "").replace(/[^a-z\s]/g, " ").replace(/\s+/g, " ").trim();
}
const explicit = new Set(["goodbye", "good bye", "bye", "bye bye", "goodnight", "good night", "see you later", "talk to you later", "end voice", "stop voice", "end the voice session", "end our conversation", "end the conversation", "stop listening", "stop listening now", "im done talking", "i am done talking", "were finished talking", "thanks goodbye", "thank you goodbye", "that will be all goodbye", "im done with this conversation"]);
const ambiguous = new Set(["im done", "i am done", "thats all", "that is all", "im good", "i am good", "thats it for now", "thatll do for now", "were good for now", "nothing else", "i dont need anything else", "thats all for now"]);
function command(text: string) {
  if (text.includes('"') || ["'", "“", "‘"].some(x => text.trim().startsWith(x))) return "";
  let value = normalize(text);
  for (const prefix of ["eri ", "eridani ", "okay ", "ok ", "please ", "you can "]) if (value.startsWith(prefix)) value = value.slice(prefix.length);
  for (const suffix of [" eri", " eridani", " thank you", " thanks", " please"]) if (value.endsWith(suffix)) value = value.slice(0, -suffix.length);
  return value;
}
export const isExplicitFarewell = (text: string) => explicit.has(command(text));
export const isVoiceEnding = isExplicitFarewell;
export function closingQuestion(text: string): ClosingQuestion | null {
  if (!text.trim().endsWith("?") || text.split("?").length !== 2) return null;
  return ({"anything else":"more", "anything else i can help you with":"more", "will that be all":"done", "will that be all for now":"done"} as const)[normalize(text) as "anything else"] ?? null;
}
export function endingDecision(text: string, context: {question?: string; age?: number; busy?: boolean; pending?: boolean; final_action_accepted?: boolean} = {}) {
  if (isExplicitFarewell(text)) return "explicit";
  if (context.busy || context.pending) return "keep";
  const value = normalize(text);
  if ((context.age ?? 0) <= 60 && ((context.question === "more" && ["no", "nope", "no thanks", "no thank you", "nope thank you"].includes(value)) || (context.question === "done" && ["yes", "yeah", "yep", "yes thanks", "yes thank you", "yep thanks"].includes(value)))) return "confirm";
  if (context.final_action_accepted && value.endsWith(" then end voice")) return "final_action";
  return ambiguous.has(command(text)) ? "offer" : "keep";
}
export type UserMarker = {sequence: number; event_id: string; end_ms: number};
export type EndCandidate = {id: string; source: UserMarker; category: string; expires_in_ms?:number};
// Observes raw provider inputs and actual local speech, never display finalization.
export class VoiceEnding {
  private seen = new Set<string>();
  private marker: UserMarker | null = null;
  private text = "";
  private lastEnd: number | null = null;
  private assistantSince = false;
  private epoch = 0;
  private inputEpoch = -1;
  private pendingEpochs:number[] = [];
  private sourceEpoch:number|null=null;
  private uncertain=false;
  private speaking = false;
  private lastActivity = 0;
  private candidateSeen = new Map<string, number>();
  speech(active: boolean, now: number) {
    if (active && !this.speaking) { this.epoch++;this.pendingEpochs.push(this.epoch);if(this.pendingEpochs.length>64){this.pendingEpochs.shift();this.uncertain=true;} }
    this.speaking = active;
    if (active) this.lastActivity = now;
  }
  captured(active:boolean, sourceEpoch:number, now:number) {
    if(this.sourceEpoch===null)this.sourceEpoch=sourceEpoch-(active?1:0);
    const changes=sourceEpoch-this.sourceEpoch;
    if(changes>64)this.uncertain=true;
    for(let i=0;i<Math.min(64,changes);i++){this.speaking=false;this.speech(true,now);}
    this.sourceEpoch=sourceEpoch;this.speech(active,now);
  }
  observe(event: Record<string, any>, now: number) {
    if (event.event_id && this.seen.has(event.event_id)) return;
    if (event.event_id) this.seen.add(event.event_id);
    if (event.type === "session.output_transcript.delta") { this.assistantSince = true; return; }
    if (event.type !== "session.input_transcript.delta" || !event.delta) return;
    const start = Number(event.start_ms ?? 0), end = Number(event.end_ms ?? start);
    const boundary=this.lastEnd === null || start-this.lastEnd>750;
    if(boundary)this.inputEpoch=this.pendingEpochs.shift() ?? this.epoch;
    if (boundary || this.assistantSince) this.text = "";
    this.text += event.delta;
    this.assistantSince = false;
    this.lastEnd = end;
    this.marker = {sequence:(this.marker?.sequence ?? 0) + 1, event_id:event.event_id ?? "", end_ms:end};
    this.lastActivity = now;
  }
  quiet(now: number, backlog = false) {
    return !this.uncertain && !!this.marker && !backlog && !this.speaking && this.epoch === this.inputEpoch && now - this.lastActivity >= 1000;
  }
  explicit(now: number, backlog = false) { return this.quiet(now, backlog) && isExplicitFarewell(this.text); }
  accepts(candidate: EndCandidate, now: number, backlog = false) {
    if (!this.candidateSeen.has(candidate.id)) this.candidateSeen.set(candidate.id, now+Math.max(0,Math.min(5000,candidate.expires_in_ms ?? 5000)));
    return this.quiet(now, backlog) && now < this.candidateSeen.get(candidate.id)! && !!this.marker &&
      candidate.source.sequence === this.marker.sequence && candidate.source.event_id === this.marker.event_id && candidate.source.end_ms === this.marker.end_ms;
  }
}
