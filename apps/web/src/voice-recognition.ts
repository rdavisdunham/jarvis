// Shared-track use requires a prior successful runtime probe, never function arity.
// Default browser-service recognition runs alongside our owned capture. onstart
// and flowing local samples together establish runtime readiness, not a constructor.
export type RecognitionAdapter = { start(track?: MediaStreamTrack): void };
export function startRecognition(
  recognition: RecognitionAdapter,
  track?: MediaStreamTrack,
  sharedTrackValidated = false,
): "shared_track" | "browser_microphone" {
  if (sharedTrackValidated && track?.readyState === "live" && track.kind === "audio") {
    recognition.start(track);
    return "shared_track";
  }
  recognition.start();
  return "browser_microphone";
}

const trackProbes = new WeakMap<object, boolean>();
// A fresh recognizer must reject an ended audio track with InvalidStateError.
// Legacy engines that ignore the argument start their service instead: abort that
// probe immediately and retain browser-microphone mode. The subsequent live
// onstart + flowing capture are still required; this probe alone is not support.
export function validatesAudioTrack(
  Type: new () => RecognitionAdapter & {abort():void},
  track?:MediaStreamTrack,
) {
  if (!track?.clone || track.kind!=="audio" || track.readyState!=="live")return false;
  if(trackProbes.has(Type))return trackProbes.get(Type)!;
  const ended=track.clone(); ended.stop();
  const recognition=new Type();let validated=false;
  try { recognition.start(ended); }
  catch(error) { validated=(error as Error).name==="InvalidStateError"; }
  finally { try { recognition.abort(); } catch { /* probe never becomes a listener */ } }
  trackProbes.set(Type,validated);return validated;
}
