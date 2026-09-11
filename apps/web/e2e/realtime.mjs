// Explicit live Realtime acceptance with a synthetic WAV, never the owner's microphone.
import { chromium, expect } from "@playwright/test";
import { readFileSync, writeFileSync } from "node:fs";
const root = new URL("../../../", import.meta.url);
const token = readFileSync(
  new URL(".runtime/pairing-code", root),
  "utf8",
).trim();
const scenario = process.env.JARVIS_VOICE_SCENARIO || "task";
if (!["task", "silence"].includes(scenario))
  throw new Error("Use task or silence as the voice scenario.");
const audioPath = new URL(
  scenario === "silence"
    ? ".runtime/voice-thanks.wav"
    : ".runtime/voice-task.wav",
  root,
);
const audioBase64 = readFileSync(audioPath).toString("base64");
const browser = await chromium.launch({
  headless: true,
  args: [
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
    "--autoplay-policy=no-user-gesture-required",
  ],
});
const context = await browser.newContext({
  permissions: ["microphone"],
  viewport: { width: 1440, height: 1000 },
});
await context.addInitScript(
  ({ audioBase64 }) => {
    let source;
    navigator.mediaDevices.getUserMedia = async () => {
      const audio = new AudioContext({ sampleRate: 48000 });
      const destination = audio.createMediaStreamDestination();
      source = audio.createBufferSource();
      const decoded = await audio.decodeAudioData(
        Uint8Array.from(atob(audioBase64), (c) => c.charCodeAt(0)).buffer,
      );
      source.buffer = window.__voiceTest.peers.length
        ? audio.createBuffer(1, 48000, 48000)
        : decoded;
      source.connect(destination);
      await audio.resume();
      return destination.stream;
    };
    window.__voiceTest = {
      peers: [],
      events: [],
      errors: [],
      audioResponses: 0,
      gateResults: [],
    };
    document.addEventListener("DOMContentLoaded", () => {
      new MutationObserver(() => {
        if (
          [...document.querySelectorAll(".message.user")].some(
            (m) =>
              m.querySelector(".transcript-status") &&
              m.querySelector("p")?.textContent !== "Listening…",
          )
        )
          window.__voiceTest.partialVisible = true;
      }).observe(document.body, {
        childList: true,
        subtree: true,
        characterData: true,
      });
    });
    window.__eventStreamOpens = 0;
    const OriginalEvents = window.EventSource;
    window.EventSource = class extends OriginalEvents {
      constructor(...args) {
        super(...args);
        this.addEventListener("open", () => window.__eventStreamOpens++);
      }
    };
    const Original = window.RTCPeerConnection;
    window.RTCPeerConnection = class extends Original {
      constructor(...args) {
        super(...args);
        window.__voiceTest.peers.push(this);
        this.addEventListener("connectionstatechange", () =>
          window.__voiceTest.events.push("peer:" + this.connectionState),
        );
        this.addEventListener("iceconnectionstatechange", () =>
          window.__voiceTest.events.push("ice:" + this.iceConnectionState),
        );
      }
      createDataChannel(...args) {
        const channel = super.createDataChannel(...args);
        channel.addEventListener("open", () => {
          if (source) {
            source.start(source.context.currentTime + 1);
            source = null;
          }
        });
        channel.addEventListener("message", (message) => {
          const event = JSON.parse(message.data);
          if (event.type === "input_audio_buffer.speech_stopped")
            window.__voiceTest.speechStoppedAt ??= Date.now();
          if (
            event.type === "conversation.item.input_audio_transcription.delta"
          )
            window.__voiceTest.firstDeltaAt ??= Date.now();
          if (
            event.type ===
            "conversation.item.input_audio_transcription.completed"
          )
            (window.__voiceTest.transcripts ??= []).push(event.transcript);
          window.__voiceTest.events.push(event.type);
          if (event.type === "error")
            window.__voiceTest.errors.push({
              code: event.error?.code,
              param: event.error?.param,
              message: event.error?.message,
            });
          if (event.type === "response.output_audio.delta")
            window.__voiceTest.audioResponses++;
          if (event.type === "response.done")
            window.__voiceTest.gateResults.push({
              status: event.response?.status,
              types: event.response?.output?.map((x) => x.type),
              text: event.response?.output
                ?.flatMap((x) => x.content || [])
                .map((x) => x.text || x.transcript || "")
                .join(""),
            });
        });
        return channel;
      }
    };
  },
  { audioBase64 },
);
const page = await context.newPage();
const evidence = { states: [], pageErrors: [], session_id: null, task: null };
const testRecovery = process.env.JARVIS_E2E_RECOVERY === "1";
let eventFaults = 0,
  pollFaults = 0;
if (testRecovery) {
  await page.route("**/api/v1/events?*", async (route) => {
    if (!eventFaults++)
      await route.fulfill({
        status: 502,
        headers: { "X-Jarvis-Test-Fault": "1" },
        body: "Synthetic proxy failure",
      });
    else await route.continue();
  });
  await page.route(/\/api\/v1\/voice\/sessions\/[^/]+$/, async (route) => {
    if (route.request().method() === "GET" && !pollFaults++)
      await route.fulfill({
        status: 502,
        contentType: "application/json",
        headers: { "X-Jarvis-Test-Fault": "1" },
        body: JSON.stringify({
          error: { code: "NETWORK", message: "Synthetic status failure" },
        }),
      });
    else await route.continue();
  });
}
page.on("pageerror", (e) => evidence.pageErrors.push(e.message));
page.on("response", async (r) => {
  if (
    !r.url().includes("/api/v1/voice/sessions") ||
    r.headers()["x-jarvis-test-fault"]
  )
    return;
  try {
    const value = await r.json();
    if (r.request().method() === "POST" && value.session_id) {
      evidence.session_id = value.session_id;
      evidence.sideband_ready = value.ready;
    }
    if (value.state && evidence.states.at(-1)?.state !== value.state)
      evidence.states.push({
        state: value.state,
        error: value.error,
        text: value.text,
      });
    if (!r.ok()) evidence.apiError = { status: r.status(), error: value.error };
  } catch {}
});
try {
  await page.goto("https://davispc.tail957c2.ts.net:9443");
  await page.getByLabel("Pairing code").fill(token);
  await page.locator(".login-card button.primary").click();
  await page
    .getByRole("button", { name: "Start a private session", exact: true })
    .click();
  const baseline = await page.evaluate(async () =>
    (await (await fetch("/api/v1/tasks")).json()).items.map((t) => t.id),
  );
  const started = Date.now();
  await page
    .getByRole("button", { name: "Talk to Eridani", exact: true })
    .click();
  const deadline = Date.now() + 65000;
  while (Date.now() < deadline) {
    const tasks = await page.evaluate(
      async () => await (await fetch("/api/v1/tasks")).json(),
    );
    const found = tasks.items?.filter(
      (t) =>
        !baseline.includes(t.id) &&
        t.title.toLowerCase().includes("acceptance voice check"),
    );
    if (scenario === "task" && found?.length) {
      evidence.task = found[0];
      evidence.task_count = found.length;
      evidence.voice_to_task_seconds = (Date.now() - started) / 1000;
    }
    if (scenario === "silence") {
      const silent = await page.evaluate(() =>
        window.__voiceTest.gateResults.some((r) => r.text.trim() === "SILENT"),
      );
      if (silent) {
        evidence.silenceGate = true;
        await new Promise((resolve) => setTimeout(resolve, 2000));
        break;
      }
    }
    if (
      scenario === "task" &&
      evidence.task &&
      (await page.evaluate(async () => {
        if (!window.__voiceTest.events.includes("output_audio_buffer.started"))
          return false;
        for (const peer of window.__voiceTest.peers)
          for (const stat of (await peer.getStats()).values())
            if (
              stat.type === "inbound-rtp" &&
              stat.kind === "audio" &&
              stat.totalAudioEnergy > 0
            )
              return true;
        return false;
      })) &&
      !evidence.interrupted
    ) {
      await page
        .getByRole("button", { name: "Stop speaking", exact: true })
        .click();
      evidence.interrupted = true;
      await new Promise((resolve) => setTimeout(resolve, 750));
      evidence.saved_after_interrupt = await page.evaluate(
        async (id) => (await fetch("/api/v1/tasks/" + id)).ok,
        evidence.task.id,
      );
      break;
    }
    if (evidence.apiError || evidence.states.at(-1)?.state === "closed") break;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  evidence.browser = await page.evaluate(async () => {
    const test = window.__voiceTest;
    const peers = [];
    for (const p of test.peers) {
      const inbound = [];
      for (const s of (await p.getStats()).values())
        if (s.type === "inbound-rtp" && s.kind === "audio")
          inbound.push({
            bytesReceived: s.bytesReceived,
            packetsReceived: s.packetsReceived,
            totalAudioEnergy: s.totalAudioEnergy,
          });
      peers.push({ connectionState: p.connectionState, inbound });
    }
    return {
      partialVisible: !!test.partialVisible,
      deltaBeforeSpeechStopped:
        !!test.firstDeltaAt &&
        (!test.speechStoppedAt || test.firstDeltaAt < test.speechStoppedAt),
      transcripts: test.transcripts,
      events: [...new Set(test.events)],
      errors: test.errors,
      results: test.gateResults,
      peers,
    };
  });
  evidence.visible_user_transcripts = await page
    .locator(".message.user")
    .allTextContents();
  evidence.user_transcript_visible = evidence.visible_user_transcripts.some(
    (t) =>
      scenario === "task"
        ? /acceptance voice check/i.test(t)
        : /thanks/i.test(t),
  );
  await page.screenshot({
    path: new URL(".runtime/voice-transcripts.png", root).pathname,
    fullPage: true,
  });
  const stop = page.getByRole("button", { name: "End voice", exact: true });
  if (await stop.count()) await stop.click();
  const audioReceived = evidence.browser.peers.some((p) =>
    p.inbound.some((s) => s.totalAudioEnergy > 0),
  );
  evidence.result =
    scenario === "silence"
      ? evidence.sideband_ready && evidence.silenceGate && !audioReceived
        ? "passed"
        : "failed"
      : evidence.sideband_ready &&
          evidence.task_count === 1 &&
          evidence.task &&
          audioReceived &&
          evidence.interrupted &&
          evidence.saved_after_interrupt &&
          evidence.browser.events.includes("output_audio_buffer.cleared")
        ? "passed"
        : "failed";
  if (scenario === "task" && evidence.result === "passed") {
    const previousSession = evidence.session_id;
    await page
      .getByRole("button", { name: "Talk to Eridani", exact: true })
      .click();
    await expect(
      page.getByRole("button", { name: "End voice", exact: true }),
    ).toBeEnabled({ timeout: 30000 });
    await expect(page.locator(".voice-panel")).toContainText("Listening", {
      timeout: 10000,
    });
    evidence.restart_passed = evidence.session_id !== previousSession;
    await page.getByRole("button", { name: "End voice", exact: true }).click();
    if (!evidence.restart_passed) evidence.result = "failed";
  }
  if (testRecovery) {
    evidence.recovery = {
      eventRequests: eventFaults,
      pollRequests: pollFaults,
      streamOpens: await page.evaluate(() => window.__eventStreamOpens),
    };
    if (eventFaults < 2 || pollFaults < 2 || !evidence.recovery.streamOpens)
      evidence.result = "failed";
  }
  if (!evidence.user_transcript_visible) evidence.result = "failed";
  if (evidence.pageErrors.length || evidence.browser?.errors.length)
    evidence.result = "failed";
} catch (error) {
  evidence.result = "failed";
  evidence.failure = error.message.split("\n")[0];
} finally {
  await page
    .evaluate(async (session) => {
      const boot = await (await fetch("/api/v1/bootstrap")).json();
      if (session)
        await fetch("/api/v1/voice/sessions/" + session + "/stop", {
          method: "POST",
          headers: {
            "X-CSRF-Token": boot.csrf,
            "Content-Type": "application/json",
          },
          body: "{}",
        });
      for (const peer of window.__voiceTest.peers) peer.close();
    }, evidence.session_id)
    .catch(() => {});
  await browser.close();
  writeFileSync(
    new URL(".runtime/realtime-" + scenario + "-evidence.json", root),
    JSON.stringify(evidence, null, 2),
  );
  console.log(JSON.stringify(evidence, null, 2));
}
if (evidence.result !== "passed") process.exitCode = 1;
