# Eridani Android app: plan for shared voice controls

Status: plan only, October 30, 2026. Nothing here is built. The web v1 PRD keeps
native Android after the release checks and the seven-day pilot
([PRD §6 exit](ERIDANI_WEB_V1_PRD.md), [TODO](TODO.md)). This plan follows the PRD:
Android reuses the tested identity, command, record-link, clarification, sync and
notification contracts. It does not rewrite them.

Target device: the owner's Pixel 10 Pro Fold (Android 16), folded and unfolded.
Android facts below were checked against platform documentation available in mid-2026.
Items marked **verify** must be confirmed in the Phase 1 spike before anyone relies on them.

## 1. Goals and non-goals

Goals

- Talk to Eri hands-free, including with the phone locked or another app open. The
  conversation uses the same GPT-Live voice, server task agent, task/reminder tools
  and Activity receipts as the website.
- Native wake word: on-device detection that keeps working when the app is not on
  screen, with a visible microphone notification and an off switch.
- Faster ways in that need no always-on mic: a Quick Settings tile, the assistant
  gesture (long-press power) and a launcher shortcut.
- The full Eridani UI on the phone without a second UI codebase.
- Reminders and question notifications as reliable as web push is today, opening
  the same deep links.

Non-goals (for this plan)

- A native Kotlin/Compose rewrite of the planner UI. The web app stays the only UI.
- Offline-first sync or a local copy of records. `sw.js` deliberately caches nothing,
  and the PRD defers offline-first sync.
- A second model runtime or a second tool path. All task work stays in the server
  chat agent (`live_voice.py`: "the existing server-side chat agent owns all task work").
- Realtime provider support. `voice_options.py` enables only `live`.
- Shared-workspace voice in the first release (see open questions).
- iOS, Wear OS, Android Auto and Home Assistant.

## 2. What exists today (the contracts to reuse)

| Area | Current implementation | What Android reuses |
| --- | --- | --- |
| Voice transport | `apps/web/src/voice.ts`: `getUserMedia` with AEC/NS/AGC, an `RTCPeerConnection` with one audio track and an `oai-events` data channel. It POSTs the offer SDP to `/api/v1/voice/sessions` and applies the returned answer. Media flows device ↔ OpenAI (CSP `connect-src https://api.openai.com`). | The same route and SDP exchange from native WebRTC. |
| Voice server | `voice.py` `start`: binds the session to `user.device_id` and a conversation owned by that device, closes any other open session on that device, reserves budget. `live_voice.py` `LiveController.connect` creates `/v1/live/sessions` with the SDP and attaches the server sideband WebSocket, which delegates to the chat agent. | Unchanged. A native client is just another device. |
| Voice lease | `GET /voice/sessions/{id}` is the lease. `Controller.watch` closes after 30 s without a poll. Web polls every 400 ms to 5 s (`POLL_*_MS`). Also `stop`, `interrupt` and `submit`. | Native polls the same status route. Hidden cadence ≤ 5 s. |
| Session end | `voice_end` tool (`voice_control.py`), client-side `VoiceEnding` (`voice-ending.ts`, farewell and closing-question logic) and `VoiceIdle` (`voice-idle.ts`, 30 s quiet, 120 s busy cap). | Port both to Kotlin and share test vectors (§5.4). |
| Wake word (web) | `wake-word.ts`: opt-in Web Speech `SpeechRecognition`, continuous, `processLocally = false` (the browser's speech service). It runs only while the page is visible (`visibilitychange` stops it). Phrases are "Eri", "Hey/Hi/OK Eri", "Eridani", plus a trailing request. `App.tsx` `wakeStart` POSTs the trailing request to `/work`, then starts voice. | The phrase grammar and the "trailing request becomes a `/work` turn" behavior. |
| Durable requests | `POST /work {turn_id, conversation_id, message, focus}`. Retrying the same `turn_id` is idempotent (guided-setup retry relies on this). | Native text/capture turns and the offline capture queue (§9). |
| App actions | Eri's `ui_*` tools (`ui_contracts.py`) dispatch through `ui_control.dispatch` to one `(owner, device)`. The device fetches them on `POST /ui/sync`, sends its `UIContext` (`ui_control.py`) and acknowledges `displayed` or `failed` within about 10 s. Context expires after 60 s. `device_bridge.py` handles cross-process work. | Native syncs as its own device, with a small UIContext and honest acknowledgements (§5.3). |
| Device preferences | `UIContext.device_preferences.wake_enabled` / `wake_supported`; the `ui_device` tool toggles them per device. | Native reports `wake_supported: true`, and "Eri, turn off the wake word" works unchanged. |
| Notifications | Web Push via `pywebpush`/VAPID (`worker.send_deliveries`). The endpoint allowlist already includes `fcm.googleapis.com`. Payload `{id,title,body,url,category,tag}`, generic text unless `detailed_notifications`. `notification_url` builds `/?view=…` deep links. `sw.js` shows the notification and opens or focuses the app on click. | Kept as-is through the TWA (§7). |
| Auth | HttpOnly `jarvis_session` cookie (SameSite=Strict, 30-day `session_hours`), `X-CSRF-Token` on writes, optional `X-Device-Id`. Sign-in uses the Google OAuth redirect (`google_routes.py`) or legacy pairing. Every `AuthSession` row has its own `device_id`. | The TWA uses cookies unchanged. Native needs a linked-device credential (§8). |
| Security headers | `api.py`: `Permissions-Policy: microphone=(self)`, a strict CSP with `frame-ancestors 'none'`. | The TWA runs in Chrome, so headers apply unchanged. A WebView shell would need changes. |

## 3. Architecture options

| | A. TWA (installed PWA in Chrome) | B. Capacitor (System WebView + plugins) | C. Native Kotlin app |
| --- | --- | --- | --- |
| UI reuse | 100%; same origin, same deploy | 100%, but the shell must load the remote URL; bundled assets would be cross-origin to the `SameSite=Strict` cookie | Rewrite; violates the PRD |
| Google sign-in | Works (Chrome, not an embedded WebView) | Google blocks OAuth in embedded WebViews: needs a Custom Tab plus a code handoff | Custom Tab or Credential Manager plus a backend token exchange |
| Notifications | Existing Web Push, delegated to the app by Chrome | No Push API in WebView: needs a new FCM sender | New FCM sender |
| Mic/voice in the UI | Existing web voice works | `getUserMedia` needs `onPermissionRequest` wiring; workable | Native |
| Hands-free / locked | Not alone: page JS stops in the background | Not alone either: the WebView is tied to an Activity | Yes |
| Native code for wake word | Yes: a TWA is an ordinary APK and can ship services | Yes (plugin) | Yes |
| Maintenance | Smallest; Digital Asset Links file | Shell, plugins, WebView quirks, FCM | Two UIs forever |

**Recommendation: A + a native voice companion in the same APK.**

- Use a Trusted Web Activity (Bubblewrap/androidx.browser) for the whole UI. It keeps
  the web app as the single UI, and today's cookies, CSRF, Google sign-in, Web Push and
  deep links work with no server changes beyond an `assetlinks.json`.
- Add a small Kotlin "voice companion" module for what the browser cannot do: a
  microphone foreground service with an on-device wake engine, a native GPT-Live
  WebRTC client against the same `/voice/sessions` route, a Quick Settings tile, the
  assistant-role entry point, and a linked-device credential.
- Capacitor's main advantage is that native code can read the WebView's cookie through
  `CookieManager`. That does not outweigh its costs: losing Google sign-in in the
  WebView, losing Web Push, and the remote-URL/cookie-origin workarounds.
- Native Kotlin UI is rejected. It is the parallel rewrite the PRD rules out.

Why native voice and not "wake word launches the web page": once the wake engine fires
in the background, Android 10+ background-activity-launch rules block opening the TWA
from a service. Since Android 14, full-screen intents are limited to calling and alarm
apps. A locked phone would also need the user to unlock it, and page JavaScript does not
run in the background. A voice session started inside the foreground service avoids all
three problems. It needs no UI, and the server already treats every device the same way.

## 4. Component overview

```
APK  app.eridani (Kotlin, minSdk 29, target = current Play requirement — verify, 36 in 2026)
├─ LauncherActivity (TWA → https://app.eridani.app/)         full UI, Web Push via Chrome
├─ VoiceCompanionService  foregroundServiceType="microphone"
│   ├─ WakeEngine (interface)  Porcupine | openWakeWord | microWakeWord
│   ├─ MicArbiter              one AudioRecord owner; pause/resume rules (§6)
│   ├─ NativeVoiceSession      WebRTC → POST /voice/sessions, status lease, stop/interrupt
│   ├─ VoiceIdle / VoiceEnding (Kotlin ports, shared fixtures)
│   └─ UiBridge                POST /ui/sync as the companion device (§5.3)
├─ VoiceOverlayActivity        small "listening / thinking / receipts" sheet; showWhenLocked policy (§8.3)
├─ AssistActivity              ACTION_ASSIST → VoiceOverlayActivity (assistant gesture)
├─ EriTileService              Quick Settings: tap = talk now; long-press = companion settings
└─ LinkedDevice                Keystore-wrapped bearer token, link/unlink
```

## 5. Reusing the API, app actions and GPT-Live voice

### 5.1 Voice session (no server change)

1. Create a conversation once per companion session (`POST /conversations`). Voice
   requires that the conversation's `device_id` match the caller's.
2. Build a `PeerConnection` with Google's WebRTC Android library (or a maintained build
   of it): one mic track (hardware AEC/NS on, `VOICE_COMMUNICATION` source) and a data
   channel named `oai-events`.
3. Wait for ICE gathering to complete, as `voice.ts` does for `live`, with a 10 s timeout.
   Then `POST /voice/sessions {conversation_id, sdp, provider:"live", voice, focus:null}`
   and apply the answer.
4. Keep the mic track disabled until the data channel opens and `session.started`
   arrives (20 s timeout). This mirrors `voice.ts` and prevents speech from being lost
   before the provider is ready.
5. Poll `GET /voice/sessions/{id}`: 400 ms while active, 1 s when stable, ≤ 5 s when
   the screen is off. The lease cutoff is 30 s. Show `receipts`, `state` and
   `awaiting_answer` in the overlay or notification.
6. End on `voice_end` (the server closes the session), on the `VoiceEnding`/`VoiceIdle`
   rules, on the notification's Stop action, or on loss of audio focus to a call (§6).
   Always `POST /stop`, then release the mic before resuming the wake engine.

Starting a native session closes any other open session on the same device. The TWA is
a different device, so web voice and native voice could both be open. `MicArbiter`
prevents that on the phone (§6).

### 5.2 Wake with a trailing request

Keep the web behavior. If the engine's post-wake capture includes a request ("Hey Eri,
remind me at five to call Sam"), send it as a `/work` turn with a stable `turn_id` before
opening the voice session. Most wake engines detect the keyword only. The first native
version therefore opens the session immediately and lets the user say the request.
Trailing-request capture becomes a later enhancement using an on-device recognizer.

### 5.3 App actions and the UI-control contract

The companion is its own device (its own `AuthSession.device_id`), so `ui_control`
already routes Eri's `ui_*` actions to it. It syncs `POST /ui/sync` at the web cadence
(`BRIDGE_ACTIVE_MS` 700 ms during voice, otherwise idle/hidden) with an honest context:

```json
{"view":"today","mobile":true,"voice_active":true,"chat_open":false,
 "device_preferences":{"wake_supported":true,"wake_enabled":true,"voice":"marin","voices":[…]}}
```

The companion handles action kinds as follows:

- **`device`** (voice, wake word, density): apply natively and acknowledge `displayed`.
- **View, record, collection and activity actions**: if the overlay is visible and the
  phone is unlocked, open the TWA at the equivalent deep link (the same `/?view=…` URLs
  `notification_url` and `task-presets.ts initialView` accept) and acknowledge
  `displayed`. Otherwise acknowledge `failed` with "The phone is locked; it's in
  Activity", so Eri says so instead of claiming it was shown. A visible activity may
  launch the TWA; a background service may not.
- **`editor`**: always `failed` ("Open Eridani to edit this"). There is no native form.

No server change is needed for the companion. Defining which `UIAction` kinds a
UI-less device accepts belongs in `ui_contracts.py` documentation, with a parity test
that lists every action kind and its native behavior so new kinds cannot be silently
dropped.

If the wake word fires while the TWA is in front, hand off instead of starting native
voice. The companion posts a `wake` message to the page over the TWA PostMessage channel
(androidx.browser, origin-validated; **verify** Chrome support in the spike). The page
runs its existing `wakeStart`, so web voice keeps full on-screen UI control. Without
PostMessage, native voice runs and on-screen actions use the deep-link path above.

### 5.4 Keeping the two clients' voice rules in step

`VoiceEnding`, `VoiceIdle` and `isWakePhrase` carry tested behavior (66 ending cases in
`voice-ending.test.ts`). Export their test vectors as JSON fixtures under
`apps/web/src/fixtures/voice/`, run them from both vitest and the Kotlin unit tests, and
add a drift guard so a web rule change without a fixture update fails CI.

## 6. Wake word, foreground service, mic lifecycle and audio focus

### 6.1 Engines (on-device; audio never leaves the phone before the wake)

| Engine | License / cost | Custom "Hey Eri" | Notes |
| --- | --- | --- | --- |
| Picovoice Porcupine | Commercial SDK; free personal tier with an AccessKey that needs periodic online validation (**verify** current terms) | Picovoice Console trains it in minutes | Best accuracy per CPU, mature Android SDK. Fastest prototype. |
| openWakeWord | Apache-2.0 code. The bundled pretrained models are CC BY-NC-SA. | Train our own with synthetic TTS data | ONNX/TFLite on Android: port the melspectrogram + embedding front end. More CPU than Porcupine. |
| microWakeWord | Apache-2.0 | Train with synthetic data (Home Assistant pipeline) | Tiny streaming TFLite models built for microcontrollers; very low CPU. Less tooling on Android. |
| Vosk keyword grammar | Apache-2.0 | Grammar only | About 40 MB model, higher CPU, more false accepts; fallback only. |
| Android `SpeechRecognizer` / system hotword | n/a | n/a | Not usable. It is not continuous, and `HotwordDetectionService` is restricted to privileged/preinstalled assistants. |

**Recommendation.** Put the engine behind a `WakeEngine` interface. Prototype with
Porcupine to measure the product (false accepts per hour, battery, latency). Then decide
between keeping Porcupine (a license question for Davin) and a custom microWakeWord or
openWakeWord model that we own. Use "Hey Eri" as the native phrase. A bare "Eri" is too
short for an always-on detector, and the web grammar's bare form would cause frequent
false accepts.

### 6.2 Foreground service rules (Android 14+)

- Declare `android:foregroundServiceType="microphone"` and the permissions
  `FOREGROUND_SERVICE`, `FOREGROUND_SERVICE_MICROPHONE`, `RECORD_AUDIO` (runtime) and
  `POST_NOTIFICATIONS` (runtime, Android 13+). Since Android 14, `startForeground` with
  the microphone type throws unless `RECORD_AUDIO` is granted at that moment.
- **While-in-use restriction:** a microphone FGS must be started while the app is
  visible or from an allowed user interaction: the tile, the notification action or the
  overlay. Never start it from `BOOT_COMPLETED`; Android 14/15 block that for the
  microphone type. After a reboot, the notification and tile say "Tap to resume
  listening". There is no silent auto-start.
- **Persistent notification** (an ongoing low-importance channel "Listening for
  Hey Eri"). Actions: *Talk now*, *Pause 1 h*, *Stop listening*. Since Android 13 the
  user can swipe it away while the service keeps running. Swiping it away counts as
  *Pause*, so a dismissed indicator never leaves a mic running unseen. The system's
  green mic indicator is shown too; that is expected and honest.
- **Doze / app standby:** an FGS process is not frozen, but network may be restricted
  during deep Doze (**verify** on Android 16). If the first voice POST after a long
  screen-off fails, retry once after `ConnectivityManager` reports a validated network.
  Ask for a battery-optimization exemption only if measurement shows it is needed; Play
  restricts that permission to qualifying use cases.
- **Play policy:** the Play Console foreground-service declaration needs the microphone
  use case and a demo video (§10).

### 6.3 Battery

Measure; don't assume. Run 8 h screen-off with listening on versus off on the Fold, using
`dumpsys batterystats` and Battery Historian, per engine. Proposed acceptance: at most
1 %/h extra drain screen-off (Davin to confirm). Levers:

- 16 kHz mono, a single `AudioRecord`, no wake lock beyond what blocking reads need.
- Pause on low battery (< 15 % or Battery Saver), at bedtime (reuse account quiet hours
  as the default listening schedule), on a phone call, and while another app records.
- Pause while Eri itself is speaking. The web already stops wake during voice; native
  does the same.

### 6.4 Mic lifecycle (`MicArbiter`)

There is one owner of capture at a time:

```
Idle ──enable──▶ Listening(wake) ──wake──▶ Releasing ──▶ VoiceSession(WebRTC mic)
  ▲                  │  ▲                                     │
  └──stop/pause──────┘  └──────── session closed + 500 ms ────┘
Listening ──call / other app captures / low battery / quiet hours──▶ Paused ──condition clears──▶ Listening
```

- Release the wake `AudioRecord` completely before WebRTC opens its own. The overlap on
  the same input breaks AEC and can silence one client.
- Concurrent capture (Android 10+): when another app records, a background capturer is
  silenced. Detect it with `AudioManager.registerAudioRecordingCallback` /
  `AudioRecordingConfiguration.isClientSilenced()`, mark the state *Paused (mic in use)*
  and stop running inference on silence.
- Calls: watch `TelephonyCallback` / audio mode `IN_CALL`/`IN_COMMUNICATION`. Pause the
  wake word, and end an active Eri session with "Paused for your call" (its accepted work
  continues, as on the web).
- Bluetooth: route the session to the active headset with
  `AudioManager.setCommunicationDevice` (Android 12+). Keep wake detection on the phone
  mic unless the user opts in to headset wake.
- Folding or unfolding the Fold must not restart capture. The service owns the session,
  not the activity. Accept a fold during a session without an audio gap.

### 6.5 Audio focus for Eri's voice

- Request `AUDIOFOCUS_GAIN_TRANSIENT` with `USAGE_ASSISTANT`/`CONTENT_TYPE_SPEECH` when
  a session starts. Music pauses or ducks and resumes after `abandonAudioFocusRequest`
  on close.
- Since Android 15, only the top app or an app running an FGS may obtain focus. The
  companion qualifies only while its microphone FGS runs, so start the FGS (even for
  tile/assist sessions with the wake word off) before requesting focus.
- Losing focus permanently (`AUDIOFOCUS_LOSS`, for example a call or another assistant)
  ends the session. A transient loss mutes playout and keeps the session until focus
  returns or the 30 s lease logic decides.
- Use `MODE_IN_COMMUNICATION` only during a session, for platform AEC. Restore the mode
  on close.

## 7. Notifications: keep Web Push first, FCM only if needed

- **Phase 1–3:** keep Web Push. In a TWA, Chrome delegates the origin's notifications to
  the app (they show under Eridani, governed by the app's `POST_NOTIFICATIONS`
  permission). Clicks open the TWA at the payload `url`, which is the existing
  `notification_url` contract. There are no server changes. The `fcm.googleapis.com`
  endpoint is already allowed, and `detailed_notifications`, quiet hours and categories
  apply unchanged.
- **Native FCM** is justified only for capabilities Web Push cannot provide:
  - native action buttons that act without opening the UI (for example *Complete* /
    *Snooze 10 min* on task alerts, calling `notification_complete` /
    `schedule_reschedule` as the linked device);
  - a high-priority data message that pre-warms the companion.
- If FCM is added, it is a second delivery channel on the same `Notification`/`Delivery`
  rows:
  - a `channel` column, defaulting to `webpush`;
  - an `fcm` sender using FCM HTTP v1 with a service-account credential in Railway
    variables;
  - the same `eligible()` gate and generic text unless `detailed_notifications`;
  - per-device dedup so a phone with both channels gets one notification.
  This is a migration plus ADDING_FEATURES §6/§7 work. It belongs in Phase 5, not earlier.

## 8. Auth and session storage

### 8.1 The TWA

The TWA uses the existing cookie session inside Chrome, unchanged. Requirement:
`/.well-known/assetlinks.json` served by the API or static host with the release (and
upload) signing-key SHA-256 fingerprints. Otherwise Chrome shows the URL bar and the app
is not a verified TWA.

### 8.2 The companion: a linked device (additive backend)

Native code cannot read Chrome's cookies, and bot keys (`bot_access.py`) are scoped
external-agent credentials without a device or voice identity. Add:

- `POST /api/v1/devices/link`: cookie-authenticated, Personal only. Returns a
  single-use code: 32 bytes, stored hashed, 2-minute expiry, rate-limited.
- `POST /api/v1/devices/link/redeem {code, label}`: public and rate-limited. Creates an
  `AuthSession` with `auth_method="android_link"` and a new `device_id`. Returns
  `{token, device_id}` once.
- `_authenticate` accepts `Authorization: Bearer <token>` only for `android_link`
  sessions. CSRF does not apply to header bearer auth, so skip it. Cookies are never
  accepted for that method, and bearer is never accepted for cookie methods.
- There is no device list today. Add `GET/DELETE /api/v1/devices` (linked sessions
  only) and a "Linked devices" list with Revoke in Settings → Privacy. Revoking deletes
  the `AuthSession` row. Decide whether unlinking Google (which deletes the owner's
  `google` sessions in `google_auth.py`) also revokes `android_link` rows; it should,
  because the link was granted from a Google-signed session. Expiry: renew while used (sliding 30 days) or re-link monthly (open question).
- Handoff from the web page in the TWA. A *Link voice companion* button in Settings →
  Voice opens an `intent://…#Intent;package=app.eridani;S.code=…;end` URL on the user's
  tap; Chrome passes it only to our package. The fallback is TWA PostMessage.

This touches routes, `access.py` and docs, and needs tests (ADDING_FEATURES §2). The
change is additive: web clients are unaffected.

### 8.3 Storage and lock-screen policy

- Store the token as an AES-GCM blob in DataStore, with the key in Android Keystore
  (`setUserAuthenticationRequired(false)`, StrongBox where available). Do not use the
  deprecated `EncryptedSharedPreferences`. Exclude it from backups
  (`dataExtractionRules`), and clear it on unlink or a 401 `NOT_AUTHORIZED`.
- Locked-phone voice is a privacy decision: Eri can read tasks aloud. The proposed
  default: when the phone is locked, the session starts in **capture-only** mode. Create
  tasks and reminders, and confirm by title only. Reading records asks for an unlock
  (`KeyguardManager.requestDismissKeyguard`). This needs a small server signal: a
  `locked` flag in the session start body or UIContext that the live instructions
  respect. That is an additive field, and older servers ignore it. Open question for
  Davin.

## 9. Offline behavior

- The UI shows a clear offline state. Nothing is cached, which matches the `sw.js`
  privacy rule and the PRD's offline-first deferral. A TWA offline fallback page
  explains this.
- Native voice needs the network (GPT-Live). On wake while offline, play a short local
  tone and show "No connection. Say it again when you're online, or type it in
  Eridani." No local speech-to-text is stored by default.
- Optional (Phase 5, opt-in): an offline **capture queue**. A typed or tile-entered
  capture is stored encrypted with a pre-generated `turn_id`, then submitted with
  `POST /work` when online. `turn_id` idempotency makes the retry safe. Do not queue
  voice audio.

## 10. Distribution

- **Sideload / internal first.** A signed APK for Davin's Fold, installed with
  `adb install`, or the Play **internal testing** track (up to 100 testers, no review
  wait, no closed-test requirement).
- **Play production** for a personal developer account requires a closed test with at
  least 12 testers for 14 days, the microphone foreground-service declaration with a
  video, a Data safety form (audio is processed for voice, not stored by the app), and a
  privacy policy URL (the public site). The current target-API requirement applies.
- **Android developer verification** (Google's 2026 rollout: apps on certified devices
  must come from verified developers, including sideloaded ones) may affect sideloading
  (**verify** status and dates). Register the package name early.
- Signing: Play App Signing with an upload key. Put both fingerprints in
  `assetlinks.json`.

## 11. Phased milestones and acceptance checks

Each phase ships only after the web v1 release checks and pilot, per the PRD.

**Phase 0: installed PWA baseline (no code).** Install the PWA on the Fold, record
voice, reminders and deep links folded and unfolded, and note what fails while locked.
*Accept:* a written baseline in `RELEASE_ACCEPTANCE.md`, which the native work must beat.

**Phase 1: TWA shell and spike.**
- Bubblewrap/androidx.browser TWA, `assetlinks.json`, internal testing.
- Spike the **verify** items: TWA PostMessage, Doze network on Android 16,
  notification delegation and the Porcupine terms.

*Accept:*
- No URL bar.
- Google sign-in, voice, Activity and web push work as in Chrome.
- A push notification opens the right `/?view=…` deep link.
- Spike findings are recorded in this document.

**Phase 2: linked device and tap-to-talk native voice.**
- Backend link/redeem/bearer.
- Kotlin `NativeVoiceSession`, Quick Settings tile, assistant role, overlay, `UiBridge`
  with the §5.3 rules, and the shared voice fixtures.

*Accept:*
- From the tile with the screen locked, "remind me tomorrow at 9 to call the dentist"
  creates the reminder with an Activity receipt (in capture-only mode if adopted).
- "Show my tasks" while locked returns the honest `failed` message; unlocked, it opens
  the TWA at Today.
- "That's all, thanks" ends the session and releases the mic: the green indicator turns
  off within 2 s.
- Revoking the device stops the next status poll with 401, and the app unlinks.
- Shared fixtures pass in both test suites.

**Phase 3: wake word foreground service.** `WakeEngine` (Porcupine prototype),
`MicArbiter`, notification actions, and the pause rules (call, other recorder, low
battery, quiet hours).

*Accept, on the Fold:*
- Wake-to-listening ≤ 1.5 s.
- False accepts ≤ 1 per 8 h of household/TV audio; true accepts ≥ 90 % at 2 m.
- Extra drain ≤ the agreed %/h screen-off.
- A call pauses and later resumes listening.
- Recording a voice memo silences the engine without errors.
- After a reboot, the engine does not listen until the user taps resume.
- Folding mid-session causes no audio gap.

**Phase 4: own the wake model (if chosen).** Train a "Hey Eri" microWakeWord or
openWakeWord model.
*Accept:* it meets or beats the Porcupine numbers on the same recorded test set, so the
license dependency can be removed.

**Phase 5: optional extras, each on its own evidence.** Native FCM notification actions,
the offline capture queue, headset wake, a home-screen widget and Play production.
*Accept per feature:* its own tests, plus no duplicate notifications when both channels
exist.

## 12. Open questions for Davin

1. Play Store or sideload only? Production needs a 12-tester, 14-day closed test.
2. Wake engine: is a Picovoice license acceptable long term, or should we budget for
   training our own model (Phase 4)?
3. Wake phrase: "Hey Eri" only on Android (recommended), or also bare "Eri" despite
   false accepts?
4. Locked-phone policy: capture-only until unlock (recommended), full access, or no
   locked use?
5. Should Eridani replace Gemini as the default assistant (long-press power) on the
   Fold, or stay tile/wake only?
6. Listening schedule: reuse notification quiet hours as "don't listen" hours by
   default?
7. Linked-device expiry: sliding renewal while used, or re-link every 30 days?
8. Shared workspaces: should the companion follow the web's current workspace, or stay
   Personal only?
9. Notification actions without opening the app (Complete/Snooze): worth adding FCM in
   Phase 5?
10. Battery budget: is ≤ 1 %/h extra drain screen-off the right bar?
