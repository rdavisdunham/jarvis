// Opt-in real API acceptance. Uses generated speech, never the owner's microphone.
import { chromium, expect } from "@playwright/test";
import { readFileSync, writeFileSync } from "node:fs";
const root = new URL("../../../", import.meta.url);
const token = readFileSync(
  new URL(".runtime/pairing-code", root),
  "utf8",
).trim();
const wav = readFileSync(new URL(".runtime/voice-task.wav", root)).toString(
  "base64",
);
const browser = await chromium.launch({
  headless: true,
  args: [
    "--use-fake-ui-for-media-stream",
    "--autoplay-policy=no-user-gesture-required",
  ],
});
const context = await browser.newContext({
  viewport: { width: 1440, height: 1000 },
  permissions: ["microphone"],
});
await context.addInitScript(
  ({ wav }) => {
    let source, audio;
    window.__live = {
      peers: [],
      events: [],
      errors: [],
      captions: { user: "", assistant: "" },
      levels: [],
    };
    navigator.mediaDevices.getUserMedia = async () => {
      audio = new AudioContext({ sampleRate: 48000 });
      const output = audio.createMediaStreamDestination();
      // Keep media frames flowing after the phrase ends, just like a live microphone.
      const silence = audio.createConstantSource();
      silence.offset.value = 0;
      silence.connect(output);
      silence.start();
      (window.__live.audioNodes ??= []).push(audio, output, silence);
      const data = Uint8Array.from(atob(wav), (c) => c.charCodeAt(0)).buffer;
      source = audio.createBufferSource();
      source.buffer = window.__live.peers.length
        ? audio.createBuffer(1, 48000, 48000)
        : await audio.decodeAudioData(data);
      source.connect(output);
      await audio.resume();
      return output.stream;
    };
    const Peer = RTCPeerConnection;
    window.RTCPeerConnection = class extends Peer {
      constructor(...args) {
        super(...args);
        window.__live.peers.push(this);
      }
      createDataChannel(...args) {
        const channel = super.createDataChannel(...args);
        let played = false;
        channel.addEventListener("message", (message) => {
          const event = JSON.parse(message.data);
          window.__live.events.push(event.type);
          if (event.type === "error")
            window.__live.errors.push(event.error?.code);
          if (event.type === "session.input_transcript.delta")
            window.__live.captions.user += event.delta;
          if (event.type === "session.output_transcript.delta")
            window.__live.captions.assistant += event.delta;
          if (event.type === "session.started" && !played) {
            played = true;
            source.start(audio.currentTime + 1);
          }
        });
        channel.addEventListener("open", () => {
          // The regression session after Live uses Realtime.
          if (window.__live.peers.length > 1 && !played) {
            played = true;
            source.start(audio.currentTime + 1);
          }
        });
        return channel;
      }
    };
  },
  { wav },
);
const page = await context.newPage();
const evidence = {
  result: "failed",
  pageErrors: [],
  apiErrors: [],
  sessions: [],
  task: null,
};
page.on("pageerror", (error) => evidence.pageErrors.push(error.message));
page.on("response", async (response) => {
  if (!response.url().includes("/api/v1/voice/sessions")) return;
  try {
    const data = await response.json();
    if (!response.ok())
      evidence.apiErrors.push({ status: response.status(), error: data.error });
    if (data.session_id) evidence.sessions.push(data.session_id);
    if (data.ui_actions?.length) evidence.ui_actions = data.ui_actions;
  } catch {}
});
try {
  await page.goto("https://davispc.tail957c2.ts.net:9443");
  await page.getByLabel("Pairing code").fill(token);
  await page.locator(".login-card button.primary").click();
  await page.getByRole("button", { name: "Open Eridani", exact: true }).click();
  await page
    .getByRole("button", { name: "Start a private session", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Voice settings", exact: true })
    .click();
  await page.getByLabel("Voice provider", { exact: true }).selectOption("live");
  await page.getByLabel("Voice", { exact: true }).selectOption("marin");
  await page.getByRole("button", {name: "Open Eridani", exact: true}).click();
  const baseline = await page.evaluate(async () =>
    (await (await fetch("/api/v1/tasks")).json()).items.map((t) => t.id),
  );
  await page
    .getByRole("button", { name: "Talk to Eridani", exact: true })
    .click();
  await expect(page.locator(".voice-panel")).toContainText(
    /Listening|Taking care/,
    { timeout: 50000 },
  );
  const deadline = Date.now() + 70000;
  while (Date.now() < deadline) {
    const tasks = await page.evaluate(
      async () => (await (await fetch("/api/v1/tasks")).json()).items,
    );
    const matches = tasks.filter(
      (t) =>
        !baseline.includes(t.id) && /acceptance voice check/i.test(t.title),
    );
    if (matches.length) {
      evidence.task = matches[0];
      evidence.task_count = matches.length;
    }
    const state = await page.evaluate(() => ({
      captions: window.__live.captions,
      events: window.__live.events,
    }));
    if (evidence.task && /added|created|saved/i.test(state.captions.assistant))
      break;
    if (evidence.apiErrors.length) break;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  await page.waitForTimeout(2500);
  evidence.browser = await page.evaluate(async () => {
    const { peers, audioNodes, ...result } = window.__live;
    result.inbound = [];
    for (const peer of peers)
      for (const stat of (await peer.getStats()).values())
        if (stat.type === "inbound-rtp" && stat.kind === "audio")
          result.inbound.push({
            totalAudioEnergy: stat.totalAudioEnergy,
            packetsReceived: stat.packetsReceived,
          });
    return result;
  });
  evidence.captionsVisible =
    (await page.locator(".message.user").count()) > 0 &&
    (await page.locator(".message.assistant").count()) > 0;
  evidence.spokenConfirmation = /added|created|saved/i.test(
    evidence.browser.captions.assistant,
  );
  evidence.voiceMode =
    (await page.locator(".app-shell.voice-active > .voice-glow").count()) === 1;
  await page.screenshot({
    path: new URL(".runtime/gpt-live.png", root).pathname,
    fullPage: true,
  });
  await page.getByRole("button", {name: "Close conversation", exact: true}).click();
  await page.setViewportSize({width: 390, height: 844});
  await expect(page.locator(".voice-dock")).toBeVisible();
  await expect(page.locator(".voice-glow")).toBeVisible();
  evidence.globalVoiceDock = true;
  await page.screenshot({path: new URL(".runtime/voice-dock-mobile.png", root).pathname, fullPage: true});
  const quietStarted = Date.now();
  await expect(page.locator(".voice-dock")).toBeHidden({timeout: 35000});
  evidence.idleStopAfterCheckSeconds = (Date.now() - quietStarted) / 1000;
  if (evidence.idleStopAfterCheckSeconds < 7) throw new Error("Voice timed out too early");
  evidence.idleTimeout = true;
  await page.setViewportSize({width: 1440, height: 1000});
  await page.getByRole("button", {name: "Open Eridani", exact: true}).click();
  await expect(page.getByLabel("Voice provider", { exact: true })).toBeEnabled({
    timeout: 20000,
  });
  evidence.gracefulClose = (
    await page.evaluate(() => window.__live.events)
  ).includes("session.closed");
  await page.getByRole("button", { name: "New chat", exact: true }).click();
  await expect(page.locator(".message")).toHaveCount(0);
  evidence.newChat = true;
  await page
    .getByLabel("Voice provider", { exact: true })
    .selectOption("realtime");
  await page
    .getByRole("button", { name: "Talk to Eridani", exact: true })
    .click();
  await expect(page.locator(".voice-panel")).toContainText("Listening", {
    timeout: 40000,
  });
  evidence.switchToRealtime = true;
  await page.getByRole("button", { name: "End voice", exact: true }).click();
  evidence.result =
    evidence.task_count === 1 &&
    /added|created|saved/i.test(evidence.browser.captions.assistant) &&
    evidence.captionsVisible &&
    evidence.voiceMode &&
    evidence.gracefulClose &&
    evidence.switchToRealtime &&
    evidence.browser.inbound.some((s) => s.totalAudioEnergy > 0) &&
    !evidence.pageErrors.length &&
    !evidence.browser.errors.length
      ? "passed"
      : "failed";
} catch (error) {
  evidence.failure = error.message.split("\n")[0];
  evidence.browser = await page
    .evaluate(() => {
      const { peers, audioNodes, ...rest } = window.__live;
      return rest;
    })
    .catch(() => ({}));
} finally {
  // Only archive the exact synthetic record created by this test.
  await page
    .evaluate(async ({ sessions, task }) => {
      const boot = await (await fetch("/api/v1/bootstrap")).json();
      const headers = {
        "Content-Type": "application/json",
        "X-CSRF-Token": boot.csrf,
      };
      for (const session of sessions)
        await fetch("/api/v1/voice/sessions/" + session + "/stop", {
          method: "POST",
          headers,
          body: "{}",
          signal: AbortSignal.timeout(20000),
        }).catch(() => {});
      if (task) {
        const current = await (await fetch("/api/v1/tasks/" + task.id)).json();
        await fetch("/api/v1/commands", {
          method: "POST",
          headers,
          body: JSON.stringify({
            command_id: crypto.randomUUID(),
            tool: "task.update",
            arguments: {
              task_id: task.id,
              expected_revision: current.revision,
              archived: true,
            },
          }),
        });
      }
    }, evidence)
    .catch(() => {});
  await browser.close();
  writeFileSync(
    new URL(".runtime/gpt-live-evidence.json", root),
    JSON.stringify(evidence, null, 2),
  );
  console.log(JSON.stringify(evidence, null, 2));
}
if (evidence.result !== "passed") process.exitCode = 1;
