import { encodeWAV } from './wavEncoder';

const STT_MODE = process.env.REACT_APP_STT_MODE || 'api';
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:3000';

let whisperPipeline = null;

// Log the configured STT mode on module load
console.log(`[STT] Mode configured: ${STT_MODE === 'browser' ? 'Browser (Transformers.js Whisper)' : 'API (Groq Whisper)'}`);


/**
 * Initialize browser-based Whisper (lazy load)
 */
async function initBrowserWhisper() {
  if (whisperPipeline) return whisperPipeline;

  console.log('[STT] Loading Whisper model in browser (first use, may take a moment)...');
  const { pipeline } = await import('@huggingface/transformers');
  whisperPipeline = await pipeline(
    'automatic-speech-recognition',
    'Xenova/whisper-tiny.en',
    { dtype: 'fp32' }  // Use fp32 for broader compatibility
  );
  console.log('[STT] Whisper model loaded successfully');
  return whisperPipeline;
}

/**
 * Transcribe audio using configured STT mode
 * @param {Float32Array} audioSamples - 16kHz audio samples
 * @returns {Promise<string|null>} - Transcribed text (browser mode) or null (API mode)
 */
export async function transcribe(audioSamples) {
  if (STT_MODE === 'browser') {
    return transcribeBrowser(audioSamples);
  } else {
    return transcribeAPI(audioSamples);
  }
}

/**
 * Browser-based transcription using Transformers.js
 */
async function transcribeBrowser(audioSamples) {
  console.log('[STT] Using BROWSER mode (Transformers.js Whisper)');
  console.log(`[STT] Processing ${audioSamples.length} audio samples locally...`);
  const transcriber = await initBrowserWhisper();
  const result = await transcriber(audioSamples);
  console.log('[STT] Browser transcription complete:', result.text.trim());
  return result.text.trim();
}

/**
 * API-based transcription - upload WAV to server
 */
async function transcribeAPI(audioSamples) {
  console.log('[STT] Using API mode (Groq Whisper)');
  const wavBlob = encodeWAV(audioSamples);
  console.log(`[STT] Uploading ${(wavBlob.size / 1024).toFixed(1)}KB WAV to server...`);
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
  // Return null - server handles transcription async
  // Client polls for result via existing mechanism
  return null;
}

/**
 * Send already-transcribed text to server (browser STT mode)
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

/**
 * Get the current STT mode
 * @returns {string} - 'api' or 'browser'
 */
export function getSTTMode() {
  return STT_MODE;
}
