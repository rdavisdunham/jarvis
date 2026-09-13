import { api, post, ApiError } from "./api";
import type { ChatMessage, UIAction, VoiceProvider } from "./types";
import { VoiceTranscript } from "./voice-transcript";
import { LiveTranscript } from "./live-transcript";
import { VoiceIdle } from "./voice-idle";
import { VoiceEnding } from "./voice-ending";
export interface VoiceState {
  state: string;
  error: string | null;
  text: string;
  closed: boolean;
  receipts: string[];
  text_id?: string;
  can_submit?: boolean;
  provider?: VoiceProvider;
  ui_actions?: UIAction[];
  idle_seconds?: number | null;
}
export class Voice {
  private pc: RTCPeerConnection | null = null;
  private audio: HTMLAudioElement | null = null;
  private stream: MediaStream | null = null;
  private session: string | null = null;
  private poll: ReturnType<typeof setTimeout> | null = null;
  private stopped = false;
  private ending = new VoiceEnding();
  private endingTimer: ReturnType<typeof setTimeout> | null = null;
  private idle = new VoiceIdle();
  private idleTimer: ReturnType<typeof setInterval> | null = null;
  private outputAnalyser: AnalyserNode | null = null;
  private inputAnalyser: AnalyserNode | null = null;
  private latestState: VoiceState = {
    state: "connecting",
    error: null,
    text: "",
    closed: false,
    receipts: [],
  };

  private disconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private pollFailures = 0;
  private transcript: VoiceTranscript;
  private liveTranscript: LiveTranscript;
  private provider: VoiceProvider = "live";
  private audioContext: AudioContext | null = null;
  private meterFrame: number | null = null;
  private stopPromise: Promise<void> | null = null;
  constructor(
    private changed: (state: VoiceState) => void,
    message: (value: ChatMessage) => void = () => {},
    private level: (value: number) => void = () => {},
  ) {
    const transcriptMessage = (value: ChatMessage) => {
      message(value);
      if (this.stopped || value.role !== "user") return;
      if (this.endingTimer) clearTimeout(this.endingTimer);
      this.endingTimer = null;
      if (this.ending.user(value)) {
        this.endingTimer = setTimeout(() => {
          this.endingTimer = null;
          if (this.stopped) return;
          this.changed({
            ...this.latestState,
            state: "closing",
            closed: false,
            error: null,
          });
          void this.stop().then(() =>
            this.changed({
              ...this.latestState,
              state: "ended",
              closed: true,
              error: null,
              idle_seconds: null,
            }),
          );
        }, 1000);
      }
    };
    this.liveTranscript = new LiveTranscript(
      "live:" + crypto.randomUUID() + ":",
      transcriptMessage,
      (value) => {
        if (this.stopped) return;
        if (value.role === "assistant")
          this.ending.assistant(value.id, value.content);
        else this.ending.user(value);
      },
    );
    this.transcript = new VoiceTranscript(
      "voice-input:" + crypto.randomUUID() + ":",
      transcriptMessage,
    );
  }
  async start(
    conversation_id: string,
    focus?: string,
    options: { provider: VoiceProvider; voice: string } = {
      provider: "live",
      voice: "marin",
    },
  ) {
    this.provider = options.provider;
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      if (this.stopped) {
        this.stream.getTracks().forEach((t) => t.stop());
        return;
      }
      this.stream.getTracks().forEach((t) => (t.enabled = false));
      this.startMeter();
      this.pc = new RTCPeerConnection();
      this.audio = new Audio();
      this.audio.autoplay = true;
      this.pc.ontrack = (event) => {
        if (this.audio) this.audio.srcObject = event.streams[0];
        if (this.audioContext) {
          this.outputAnalyser = this.audioContext.createAnalyser();
          this.outputAnalyser.fftSize = 512;
          this.audioContext
            .createMediaStreamSource(event.streams[0])
            .connect(this.outputAnalyser);
        }
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
      let liveStarted = false;
      const liveReady = new EventTarget();
      channel.addEventListener("message", (message) => {
        if (this.stopped) return;
        try {
          const event = JSON.parse(message.data);
          if (event.type === "session.started") {
            liveStarted = true;
            liveReady.dispatchEvent(new Event("ready"));
          }
          const now = performance.now();
          if (
            event.type === "input_audio_buffer.speech_started" &&
            this.endingTimer
          ) {
            clearTimeout(this.endingTimer);
            this.endingTimer = null;
          }
          if (event.type === "input_audio_buffer.speech_started")
            this.idle.speech(true, now);
          if (event.type === "input_audio_buffer.speech_stopped")
            this.idle.speech(false, now);
          if (
            [
              "session.input_transcript.delta",
              "session.output_transcript.delta",
              "output_audio_buffer.started",
              "output_audio_buffer.stopped",
              "output_audio_buffer.cleared",
            ].includes(event.type)
          )
            this.idle.touch(now);
          if (this.provider === "live") this.liveTranscript.receive(event);
          else this.transcript.receive(event);
        } catch {
          /* Ignore malformed provider events. */
        }
      });
      await this.pc.setLocalDescription(await this.pc.createOffer());
      if (
        this.provider === "live" &&
        this.pc.iceGatheringState !== "complete"
      ) {
        await new Promise<void>((resolve, reject) => {
          const peer = this.pc!;
          const changed = () => {
            if (peer.iceGatheringState === "complete") finish();
          };
          const finish = (error?: Error) => {
            clearTimeout(timeout);
            peer.removeEventListener("icegatheringstatechange", changed);
            if (error) reject(error);
            else resolve();
          };
          const timeout = setTimeout(
            () =>
              finish(new Error("Network setup timed out. Try voice again.")),
            10000,
          );
          peer.addEventListener("icegatheringstatechange", changed);
          changed();
        });
      }
      if (this.stopped) return;
      const data = await post<{
        session_id: string;
        sdp: string;
        ready: boolean;
      }>("/voice/sessions", {
        conversation_id,
        ...options,
        focus: focus ?? null,
        sdp: this.pc.localDescription!.sdp,
      });
      this.session = data.session_id;
      if (this.stopped) {
        await post("/voice/sessions/" + data.session_id + "/stop").catch(
          () => {},
        );
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
      if (this.provider === "live" && !liveStarted) {
        await new Promise<void>((resolve, reject) => {
          const finish = () => {
            clearTimeout(timeout);
            resolve();
          };
          const timeout = setTimeout(() => {
            liveReady.removeEventListener("ready", finish);
            reject(
              new Error(
                "GPT-Live did not become ready. Try Realtime or start again.",
              ),
            );
          }, 20000);
          liveReady.addEventListener("ready", finish, { once: true });
        });
      }
      if (this.stopped) return;
      this.stream.getTracks().forEach((t) => (t.enabled = true));
      this.idle.start(performance.now());
      this.idleTimer = setInterval(() => this.checkIdle(), 200);
      this.check();
    } catch (error) {
      await this.stop();
      throw error;
    }
  }
  private check = async () => {
    if (!this.session || this.stopped) return;
    const session = this.session;
    try {
      const state = await api<VoiceState>("/voice/sessions/" + session);
      if (this.stopped || this.session !== session) return;
      this.pollFailures = 0;
      this.latestState = state;
      if (this.provider === "realtime" && state.text_id && state.text)
        this.ending.assistant(state.text_id, state.text);
      this.idle.state(state.state, performance.now());
      this.changed({
        ...state,
        idle_seconds: this.idle.remaining(performance.now()),
      });
      if (state.closed) {
        await this.stop();
        return;
      }
    } catch (error) {
      if (this.stopped || this.session !== session) return;
      const terminal =
        error instanceof ApiError &&
        ["NOT_FOUND", "NOT_AUTHORIZED"].includes(error.code);
      if (!terminal && ++this.pollFailures <= 3) {
        this.poll = setTimeout(this.check, 800);
        return;
      }
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
    if (!this.stopped) this.poll = setTimeout(this.check, 400);
  };
  private startMeter() {
    if (typeof AudioContext === "undefined" || !this.stream) return;
    try {
      const context = new AudioContext();
      this.audioContext = context;
      void context.resume().catch(() => {});
      const analyser = context.createAnalyser();
      analyser.fftSize = 256;
      this.inputAnalyser = analyser;
      context.createMediaStreamSource(this.stream).connect(analyser);
      const samples = new Uint8Array(analyser.fftSize);
      const tick = () => {
        if (this.stopped) return;
        analyser.getByteTimeDomainData(samples);
        const rms = Math.sqrt(
          samples.reduce(
            (sum, sample) => sum + ((sample - 128) / 128) ** 2,
            0,
          ) / samples.length,
        );
        this.level(Math.min(1, rms * 7));
        this.meterFrame = requestAnimationFrame(tick);
      };
      tick();
    } catch {
      /* Audio metering is decorative; voice still works. */
    }
  }
  private checkIdle() {
    if (this.stopped) return;
    const now = performance.now();
    // Sample incoming playout as well as microphone energy: output text can arrive
    // seconds before the corresponding speech has finished playing.
    for (const analyser of [this.outputAnalyser, this.inputAnalyser]) {
      if (!analyser) continue;
      const samples = new Uint8Array(analyser.fftSize);
      analyser.getByteTimeDomainData(samples);
      const rms = Math.sqrt(
        samples.reduce((sum, x) => sum + ((x - 128) / 128) ** 2, 0) /
          samples.length,
      );
      if (rms > 0.012) this.idle.touch(now);
    }
    if (this.idle.expired(now)) {
      this.changed({
        ...this.latestState,
        state: "closing",
        closed: false,
        idle_seconds: 0,
      });
      void this.stop().then(() =>
        this.changed({
          ...this.latestState,
          state: "idle_timeout",
          closed: true,
          error: null,
          idle_seconds: 0,
        }),
      );
    } else {
      this.changed({
        ...this.latestState,
        idle_seconds: this.idle.remaining(now),
      });
    }
  }
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
    if (this.stopPromise) return this.stopPromise;
    this.stopPromise = this.cleanup();
    return this.stopPromise;
  }
  private async cleanup() {
    this.stopped = true;
    if (this.endingTimer) clearTimeout(this.endingTimer);
    this.endingTimer = null;
    if (this.idleTimer) clearInterval(this.idleTimer);
    this.idleTimer = null;
    this.liveTranscript.dispose();
    if (this.meterFrame !== null) cancelAnimationFrame(this.meterFrame);
    this.meterFrame = null;
    void this.audioContext?.close().catch(() => {});
    this.audioContext = null;
    this.level(0);
    if (this.poll) clearTimeout(this.poll);
    if (this.disconnectTimer) clearTimeout(this.disconnectTimer);
    this.disconnectTimer = null;
    // Mute immediately, but keep Live's negotiated media track alive through finalization.
    this.stream?.getTracks().forEach((t) => {
      t.enabled = false;
      if (this.provider !== "live") t.stop();
    });
    if (this.audio) {
      this.audio.pause();
      this.audio.srcObject = null;
    }
    this.audio = null;
    const session = this.session;
    this.session = null;
    // Keep the Live media connection alive until the server collects session.closed usage.
    if (session && this.provider === "live") {
      try {
        await api("/voice/sessions/" + session + "/stop", {
          method: "POST",
          body: "{}",
          signal: AbortSignal.timeout(15000),
        });
      } catch {
        /* The server lease finalizes abandoned sessions. */
      }
    }
    if (this.provider === "live")
      this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    if (this.pc) {
      this.pc.onconnectionstatechange = null;
      this.pc.close();
    }
    this.pc = null;
    if (session && this.provider !== "live") {
      try {
        await post("/voice/sessions/" + session + "/stop");
      } catch {
        /* Server lease closes orphan sessions. */
      }
    }
  }
}
