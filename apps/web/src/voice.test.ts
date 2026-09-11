import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { Voice } from "./voice";
import { api, post } from "./api";

vi.mock("./api", () => ({ api: vi.fn(), post: vi.fn() }));

class Channel extends EventTarget {
  readyState = "open";
}
class Peer {
  static current: Peer;
  connectionState = "connected";
  onconnectionstatechange: (() => void) | null = null;
  ontrack: unknown;
  localDescription = { sdp: "synthetic" };
  constructor() {
    Peer.current = this;
  }
  addTrack() {}
  channel = new Channel();
  createDataChannel() {
    return this.channel;
  }
  async createOffer() {
    return { sdp: "synthetic" };
  }
  async setLocalDescription() {}
  async setRemoteDescription() {}
  close = vi.fn();
  change(state: string) {
    this.connectionState = state;
    this.onconnectionstatechange?.();
  }
}

let stopped = vi.fn();
beforeEach(() => {
  vi.useFakeTimers();
  stopped = vi.fn();
  const track = { enabled: true, stop: stopped };
  vi.stubGlobal("navigator", {
    mediaDevices: {
      getUserMedia: vi.fn().mockResolvedValue({
        getTracks: () => [track],
        getAudioTracks: () => [track],
      }),
    },
  });
  vi.stubGlobal("RTCPeerConnection", Peer);
  vi.stubGlobal(
    "Audio",
    class {
      autoplay = false;
      srcObject = null;
      pause() {}
    },
  );
  vi.mocked(post).mockResolvedValue({
    session_id: "test-session",
    sdp: "answer",
    ready: true,
  });
  vi.mocked(api).mockResolvedValue({
    state: "listening",
    error: null,
    text: "",
    closed: false,
    receipts: [],
  });
});
afterEach(() => {
  vi.clearAllTimers();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

test("brief disconnection recovers without ending the voice session", async () => {
  const changed = vi.fn();
  const voice = new Voice(changed);
  await voice.start("conversation");
  Peer.current.change("disconnected");
  await vi.advanceTimersByTimeAsync(3000);
  expect(stopped).not.toHaveBeenCalled();
  Peer.current.change("connected");
  await vi.advanceTimersByTimeAsync(3000);
  expect(stopped).not.toHaveBeenCalled();
  expect(changed.mock.calls.some(([state]) => state.closed)).toBe(false);
  await voice.stop();
});

test("persistent disconnection releases microphone and ends the server session", async () => {
  const changed = vi.fn();
  const voice = new Voice(changed);
  await voice.start("conversation");
  Peer.current.change("disconnected");
  await vi.advanceTimersByTimeAsync(5000);
  expect(stopped).toHaveBeenCalledOnce();
  expect(Peer.current.close).toHaveBeenCalledOnce();
  expect(changed).toHaveBeenCalledWith(
    expect.objectContaining({ state: "disconnected", closed: true }),
  );
  expect(post).toHaveBeenCalledWith("/voice/sessions/test-session/stop");
});

test("capture and listening wait for the provider data channel", async () => {
  const changed = vi.fn();
  const voice = new Voice(changed);
  let resolveSession: (value: unknown) => void = () => {};
  vi.mocked(post).mockReturnValueOnce(
    new Promise((resolve) => {
      resolveSession = resolve;
    }),
  );
  const starting = voice.start("conversation");
  await vi.advanceTimersByTimeAsync(0);
  Peer.current.channel.readyState = "connecting";
  resolveSession({ session_id: "test-session", sdp: "answer", ready: true });
  await vi.advanceTimersByTimeAsync(0);
  const stream = await vi.mocked(navigator.mediaDevices.getUserMedia).mock
    .results[0].value;
  expect(stream.getTracks()[0].enabled).toBe(false);
  expect(api).not.toHaveBeenCalled();
  Peer.current.channel.readyState = "open";
  Peer.current.channel.dispatchEvent(new Event("open"));
  await starting;
  expect(stream.getTracks()[0].enabled).toBe(true);
  expect(api).toHaveBeenCalled();
  await voice.stop();
});
