// The deadline follows actual playout, not the arrival of generated text.
export class VoiceIdle {
  private lastActivity = 0;
  private active = false;
  private busy = false;
  private speaking = false;
  constructor(private timeout = 30000) {}
  start(now: number) {
    this.active = true;
    this.touch(now);
  }
  touch(now: number) {
    this.lastActivity = now;
  }
  state(state: string, now: number) {
    const busy = [
      "connecting",
      "closing",
      "thinking",
      "evaluating",
      "acting",
      "working",
      "speaking",
    ].includes(state);
    if (this.busy && !busy) this.touch(now);
    this.busy = busy;
  }
  speech(active: boolean, now: number) {
    this.speaking = active;
    this.touch(now);
  }
  remaining(now: number) {
    return !this.active || this.busy || this.speaking
      ? null
      : Math.max(
          0,
          Math.ceil((this.timeout - (now - this.lastActivity)) / 1000),
        );
  }
  expired(now: number) {
    return this.remaining(now) === 0;
  }
}
