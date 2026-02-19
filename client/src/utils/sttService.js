import { encodeWAV } from './wavEncoder';

const API_URL = `${window.location.protocol}//${window.location.hostname}:3000`;

/**
 * Transcribe audio by uploading WAV to server (Groq Whisper API).
 * Used as fallback when STT_PROVIDER is not 'local'.
 * @param {Float32Array} audioSamples - 16kHz audio samples
 * @returns {Promise<null>} - Server handles transcription async
 */
export async function transcribe(audioSamples) {
  const wavBlob = encodeWAV(audioSamples);
  console.log(`[STT] Uploading ${(wavBlob.size / 1024).toFixed(1)}KB WAV to server (Groq Whisper)...`);
  const formData = new FormData();
  formData.append('audio', wavBlob, 'recording.wav');

  const response = await fetch(`${API_URL}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Upload failed');
  }

  console.log('[STT] Audio uploaded, server will transcribe via Groq API');
  return null;
}

/**
 * Send already-transcribed text to server
 * @param {string} text - Transcribed text to send
 * @param {boolean} enableTts - Whether to enable TTS for the response
 */
export async function sendTranscribedText(text, enableTts = true) {
  console.log(`[STT] Sending transcribed text to server (TTS: ${enableTts ? 'enabled' : 'disabled'})`);
  const response = await fetch(`${API_URL}/text-message`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: text, enableTts }),
  });

  if (!response.ok) {
    throw new Error('Failed to send text');
  }
}
