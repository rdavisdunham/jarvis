import { api, post } from "./api";
export interface VoiceState {
  state: string;
  error: string | null;
  text: string;
  closed: boolean;
  receipts: string[];
}
export class Voice {
  private pc: RTCPeerConnection | null = null;
  private audio: HTMLAudioElement | null = null;
  private stream: MediaStream | null = null;
  private session: string | null = null;
  private poll: ReturnType<typeof setTimeout> | null = null;
  private stopped = false;
  private disconnectTimer: ReturnType<typeof setTimeout> | null = null;
  constructor(private changed: (state: VoiceState) => void) {}
  async start(conversation_id: string, focus?: string) {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      this.stream.getTracks().forEach((t) => (t.enabled = false));
      this.pc = new RTCPeerConnection();
      this.audio = new Audio();
      this.audio.autoplay = true;
      this.pc.ontrack = (event) => {
        if (this.audio) this.audio.srcObject = event.streams[0];
      };
      this.pc.onconnectionstatechange = () => {
        const state = this.pc?.connectionState;
        if (state === "connected" && this.disconnectTimer) {
          clearTimeout(this.disconnectTimer);
          this.disconnectTimer = null;
        }
        const endDisconnectedSession = () => {
          this.changed({
            state: "disconnected",
            error: "Voice disconnected. Your saved work is safe.",
            text: "",
            closed: true,
            receipts: [],
          });
          void this.stop();
        };
        if (state === "failed") endDisconnectedSession();
        // ICE can briefly disconnect during route changes. Give it time to recover.
        if (state === "disconnected" && !this.disconnectTimer) {
          this.disconnectTimer = setTimeout(() => {
            this.disconnectTimer = null;
            if (this.pc?.connectionState === "disconnected")
              endDisconnectedSession();
          }, 5000);
        }
      };
      this.pc.addTrack(this.stream.getAudioTracks()[0], this.stream);
      const channel = this.pc.createDataChannel("oai-events");
      await this.pc.setLocalDescription(await this.pc.createOffer());
      const data = await post<{
        session_id: string;
        sdp: string;
        ready: boolean;
      }>("/voice/sessions", {
        conversation_id,
        focus: focus ?? null,
        sdp: this.pc.localDescription!.sdp,
      });
      this.session = data.session_id;
      if (this.stopped) {
        await this.stop();
        return;
      }
      await this.pc.setRemoteDescription({ type: "answer", sdp: data.sdp });
      if (!data.ready) throw new Error("Voice controls are not ready.");
      // SDP negotiation can finish before the provider can receive speech.
      // Keep capture disabled and the UI connecting until its data channel opens.
      await new Promise<void>((resolve, reject) => {
        if (channel.readyState === "open") {
          resolve();
          return;
        }
        const finish = (error?: Error) => {
          clearTimeout(timeout);
          channel.removeEventListener("open", opened);
          channel.removeEventListener("close", closed);
          channel.removeEventListener("error", closed);
          if (error) reject(error);
          else resolve();
        };
        const opened = () => finish();
        const closed = () =>
          finish(new Error("Voice could not connect. Start a new session."));
        const timeout = setTimeout(closed, 15000);
        channel.addEventListener("open", opened);
        channel.addEventListener("close", closed);
        channel.addEventListener("error", closed);
      });
      if (this.stopped) return;
      this.stream.getTracks().forEach((t) => (t.enabled = true));
      this.check();
    } catch (error) {
      await this.stop();
      throw error;
    }
  }
  private check = async () => {
    if (!this.session || this.stopped) return;
    try {
      const state = await api<VoiceState>("/voice/sessions/" + this.session);
      this.changed(state);
      if (state.closed) {
        await this.stop();
        return;
      }
    } catch {
      this.changed({
        state: "disconnected",
        error: "Voice control disconnected. Start a new session.",
        text: "",
        closed: true,
        receipts: [],
      });
      await this.stop();
      return;
    }
    this.poll = setTimeout(this.check, 400);
  };
  async interrupt() {
    if (this.audio) {
      this.audio.muted = true;
      setTimeout(() => {
        if (this.audio) this.audio.muted = false;
      }, 400);
    }
    if (this.session)
      await post("/voice/sessions/" + this.session + "/interrupt");
  }
  async submit() {
    if (this.session) await post("/voice/sessions/" + this.session + "/submit");
  }
  async stop() {
    this.stopped = true;
    if (this.poll) clearTimeout(this.poll);
    if (this.disconnectTimer) clearTimeout(this.disconnectTimer);
    this.disconnectTimer = null;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    if (this.audio) {
      this.audio.pause();
      this.audio.srcObject = null;
    }
    this.audio = null;
    if (this.pc) {
      this.pc.onconnectionstatechange = null;
      this.pc.close();
    }
    this.pc = null;
    const session = this.session;
    this.session = null;
    if (session) {
      try {
        await post("/voice/sessions/" + session + "/stop");
      } catch {
        /* Server lease closes orphan sessions. */
      }
    }
  }
}
