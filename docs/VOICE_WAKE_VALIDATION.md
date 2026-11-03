# Foreground voice and wake validation

Approved plan: task 1f324cd9-aed5-4c14-83f4-8f0a6de456b3, revision 1.
Baseline: clean 1d8eb83. Deployment is not authorized.

Ordinary gratitude keeps listening. Uncertain endings stay open. Clear standalone
farewells close after continued-speech cancellation; contextual endings require a
fresh, session-bound backend candidate. Closing offers require an ambiguous ending,
no pending work/question, and a 180-second cooldown; observed reply context lasts
60 seconds. Quiet timeout is 120 seconds, separate from the 120-second busy cap and
30-second missing-client lease. Accepted durable work survives voice closure.

Audio contract: one memory-only capture, three seconds rolling history, utterance
pin with 200 ms pre-roll, 60 seconds / 16 MiB maximum. One sample cursor feeds one
derived media track at original rate. Only confirmed wake/manual start creates a
paid provider session. Whole startup deadline is 30 seconds. No transcript-to-work
submission runs beside the audio handoff. Browser speech recognition uses the
browser speech service; standby is not described as offline. Constructor presence
does not prove capture/recognition coexistence or shared-track recognition.

Timing boundaries (content-free, local monotonic clocks only): detection, capture,
preparation, SDP, create request/response, sideband, data channel, session.started,
first sent sample, drain complete, first audible output. Client/server durations
must be reported separately. No microphone samples are persisted or logged.

## Evidence worksheet

This worker has a Linux shell and deterministic test fakes. No physical target,
consented speech samples, live-provider audio, or matched baseline latency samples
are available. Exact installed versions must be read on each actual device; a
current release number from the internet is not an installed-device measurement.

| Target / configuration | Exact Chrome / OS | Wake | First request / cue | Ten cycles | Cold / returning timing |
| --- | --- | --- | --- | --- | --- |
| Pixel 10 Pro Fold, folded, speaker | unavailable | not run | not run | not run | not run |
| Pixel 10 Pro Fold, unfolded, speaker | unavailable | not run | not run | not run | not run |
| Pixel, available headphones | unavailable | not run | not run | not run | not run |
| Windows desktop, speaker / available headphones | unavailable | not run | not run | not run | not run |
| Surface laptop, speaker / available headphones | unavailable | not run | not run | not run | not run |
| macOS Chrome, speaker / available headphones | unavailable | not run | not run | not run | not run |
| iOS Chrome, speaker / available headphones | unavailable | not run | not run | not run | not run |

For each row record speaker, ambient conditions, exact OS/Chrome, 10 bare Eri and
10 Hey Eri attempts (at least five immediate commands), 20 negative utterances and
10 wake/end cycles. Floor: 19/20 wakes without repetition, zero negative activations,
zero lost/duplicate first requests and zero cues over speech. Safari or viewport
simulation cannot establish Chrome iOS support; tap-to-talk does not pass wake.
Any demonstrated incompatibility remains unresolved.

Collect at least 10 matched cold and 10 returning samples on the same network per
available device. Report recognition, confirmed wake-to-ready, first response and
drain p50/p95 separately. Dispatch target <=200 ms; ready-to-first-delivery <=100 ms.
Median ready target: 20% reduction vs baseline, p95 no more than 10% worse. Baseline
and revised physical/provider measurements are pending; synthetic checks cannot
establish these thresholds.

## Automated verification

Required record: Eridani target `jarvis`, verification `full`, for the final revision.
The repository path runs pytest, TypeScript and Vitest offline with prepared
dependencies and isolated PostgreSQL. Record and file map will be completed with
implementation. No physical or latency acceptance is claimed by a passing suite.

## Contracts and deterministic coverage

- `tests/fixtures/voice-ending.json`: shared Python/TypeScript decisions, including
  gratitude, both closing polarities, uncertainty, quotes and final actions.
- `voice_end_policy.py`, `work_intake.py`, `work_runner.py`, `live_voice.py` and
  `voice.py`: source metadata in encrypted payloads, candidate proposal/publication,
  observed approved questions, cooldown, expiry and lock-protected confirmation.
- `voice-ending.ts`: raw provider input marker and local speech epoch; display
  caption finalization cannot authorize closure.
- `voice-idle.ts`: quiet window, busy cap and startup drain; silent samples are not
  activity. Existing backend client lease remains 30 seconds.
- `voice-audio.ts` and `voice-worklet.ts`: shared real processor core, original-rate
  sample cursor, conservative silence compaction and explicit bounds. Vitest loads
  the actual worklet adapter with AudioWorklet globals supplied by the harness.
- `voice-capture.ts`, `voice-recognition.ts`, `wake-word.ts`, `voice-coordinator.ts`:
  foreground capture/recognition readiness, stable interim prefix, alternatives,
  generations, recovery and account/device preference. Shared-track mode is used
  only when runtime-validated; the default service recognizer uses its microphone
  alongside owned capture. Physical coexistence has not been measured.

At the pre-integration checkpoint, TypeScript and all 516 web tests passed (one
pre-existing disabled Realtime test skipped). Queue fixtures cover 0/2/5/15-second
setup delay, uninterrupted append/read, safe silence, boundaries, 60-second/byte
limits, negative recognition release, reset, overflow and stale worklet commands.
Backend subsets passed 96 and then 73 tests as the policy developed; these are
local diagnostic runs, not Eridani's required final full record.

## Integration and recovery

`App.tsx` owns one coordinator bound to account, device and workspace. It creates
no paid session for wake standby. `Voice.start` receives the existing capture and
prepares local SDP while conversation/UI preparation runs. Only the single derived
media track delivers the selected utterance after data-channel/session readiness;
the former wake suffix `/work` submission is removed. No PCM append is sent over
the Live data channel. `live_http.py` lazily reuses bounded HTTP connections and the
application lifespan closes the pool after controller settlement.

`voice-startup-cue.ts` requests one cue only for positive wake-only speech, delivery,
readiness and current quiet. A worklet acknowledgment rechecks quiet before that
request. During cue barge-in, the worklet holds the first voiced frame while the
player discards stale output; then the same queue resumes in order. This gate is
disarmed for ordinary full-duplex conversation. A quiet Live greeting may be
paraphrased or absent, so the fallback uses actual output activity, not an append
receipt. These controls follow the [Live session guide](https://developers.openai.com/api/docs/guides/live-conversations)
and [application playback guidance](https://developers.openai.com/api/docs/guides/voice-server-controls?api=live).
Provider audio and cancellation behavior still require physical/live validation.

An ambiguous VAD/recognition selection, lost onset, overflow, suspended context,
ended microphone or 30-second startup deadline fails visibly and clears the local
queue. It never restores the old parallel text route or automatically replays
uncertain audio. Check Activity for work already accepted before a failure. Late
provider creation is attached and closed through a bounded cleanup path without
reviving the cancelled controller. Final usage remains known or explicitly
uncertain under the existing settlement rules.

Manual End stops local transmission/playback before waiting for graceful provider
settlement. Wake rearming waits for detachment and settlement; if opted in, the
microphone can then remain in use for wake. Turning wake off releases standby
capture; during established voice it disables future wake and leaves End Voice
available. Permission denial does not loop prompts. Account/workspace/logout and
cross-tab account changes clear old capture, buffers and callbacks before rebinding.

## Measurement status

| Evidence | Baseline 1d8eb83 | Revised implementation |
| --- | --- | --- |
| Browser recognition | Final-only; requires the whole command to finalize | Stable 200 ms prefix or final, up to three alternatives; physical accuracy pending |
| Audio startup | Fresh mic disabled until ready; suffix text submitted separately | Owned capture, one sample cursor/track; 0/2/5/15-second simulated setup tests pass |
| Standby cost | Browser speech service | Browser speech service plus bounded local capture; no speculative Live create |
| Recognition / wake-to-ready / response / drain p50 and p95 | Not measured on available target hardware | Not measured on available target hardware |
| >=19/20 wakes, zero scripted negatives | Not run | Not run |
| No loss/duplicate request and no cue over speech on target devices | Not run | Deterministic contracts pass; physical/provider acceptance not run |
| Ten physical cycles without stuck microphone | Not run | Ten-cycle coordinator fake passes; physical acceptance not run |

Client `voiceDiagnostics()` retains at most 256 content-free stage events, keyed by
opaque attempt ID: detection, capture, app preparation, SDP, create request/response,
data channel, session.started, first sent samples, drain and first active output.
Counters cover detection, duplicate suppression, recovery, overflow and stale
ending rejection. Backend `live_create` uses server-local monotonic durations; sideband/session
events provide UTC correlation. Client/server clocks must not
be subtracted from each other. Acoustic onset, actual listening accuracy and the
20% median / 10% p95 improvement targets need the worksheet's matched samples;
none is established by renaming stages or by fake timers.

Validation host: Linux 6.18.33.2-microsoft-standard-WSL2 x86_64, Python 3.12.3,
Node v22.23.2. This is not a Windows Chrome device result. The production Vite build
was checked separately and includes the worklet; it emits the existing large-chunk
advisory. Catalog validation/coverage ratchet and CI's Python correctness checks
pass. New acceptance protocols remain unrun and do not gain a physical pass from
supporting suites.

The required Eridani `jarvis` / `full` record is still pending runner execution for
the completed branch. Local full-path output is diagnostic evidence and does not
substitute for that record. No deployment, paid inference or physical-device pass
is claimed.

Local verification on 2026-11-03: the full prepared-dependency path completed with
1,925 pytest passes (one existing skip), TypeScript passing, and 536 Vitest passes
(one existing disabled Realtime skip). The subsequent remaining-TTL status field
was checked with 84 backend policy/startup/controller tests and the complete web
path (TypeScript plus 537 Vitest passes, one existing skip). That small follow-up
still needs Eridani's fresh full record on the final commit. All local checks used
synthetic provider fakes and the isolated test PostgreSQL service.
