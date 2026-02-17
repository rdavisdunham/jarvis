/**
 * AudioWorklet processor: captures mic audio, converts Float32 → Int16 PCM,
 * and posts chunks to the main thread for WebSocket transmission.
 *
 * Runs in the audio thread for consistent ~100ms chunks at 16kHz.
 * Input is typically 48kHz from the browser mic; we downsample to 16kHz.
 */

class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = [];
    this._targetSampleRate = 16000;
    // sampleRate is a global in AudioWorkletGlobalScope
    this._inputSampleRate = sampleRate;
    this._chunkSamples = this._targetSampleRate * 0.1; // 100ms = 1600 samples at 16kHz
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channelData = input[0]; // mono

    // Downsample if needed
    const ratio = this._inputSampleRate / this._targetSampleRate;
    if (ratio > 1) {
      for (let i = 0; i < channelData.length; i += ratio) {
        this._buffer.push(channelData[Math.floor(i)]);
      }
    } else {
      for (let i = 0; i < channelData.length; i++) {
        this._buffer.push(channelData[i]);
      }
    }

    // Send chunks of ~100ms
    while (this._buffer.length >= this._chunkSamples) {
      const chunk = this._buffer.splice(0, this._chunkSamples);
      const int16 = new Int16Array(chunk.length);
      for (let i = 0; i < chunk.length; i++) {
        const s = Math.max(-1, Math.min(1, chunk[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      this.port.postMessage(int16.buffer, [int16.buffer]);
    }

    return true;
  }
}

registerProcessor('pcm-processor', PCMProcessor);
