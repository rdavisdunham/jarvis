/**
 * StreamingAudioPlayer — Web Audio API streaming PCM playback with interrupt.
 *
 * Uses AudioBufferSourceNode scheduling for gapless playback of int16 PCM
 * chunks arriving over WebSocket at ~250ms intervals.
 */

export class StreamingAudioPlayer {
  constructor() {
    this.audioContext = null;
    this.sampleRate = 24000;
    this.nextPlayTime = 0;
    this.isPlaying = false;
    this.scheduledSources = [];
    this.onPlaybackEnd = null;
    this._endTimeout = null;
    this._lastChunkDuration = 0;
    this._chunkBuffer = [];
    this._playbackStarted = false;
    this._minBufferChunks = 3; // Buffer 3 chunks (~750ms) before starting playback
  }

  /**
   * Initialize a new stream. Call when audio_start is received.
   */
  startStream(responseId, sampleRate) {
    this.dispose();
    this.sampleRate = sampleRate || 24000;
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: this.sampleRate,
    });
    this.isPlaying = true;
    this.scheduledSources = [];
    this._lastChunkDuration = 0;
    this._chunkBuffer = [];
    this._playbackStarted = false;
  }

  /**
   * Queue a PCM int16 chunk for gapless playback.
   * @param {ArrayBuffer} pcmInt16Data — raw int16 PCM bytes (no header)
   */
  queueChunk(pcmInt16Data) {
    if (!this.audioContext || !this.isPlaying) return;

    if (!this._playbackStarted) {
      // Buffering phase: accumulate chunks before starting playback
      this._chunkBuffer.push(pcmInt16Data);
      if (this._chunkBuffer.length >= this._minBufferChunks) {
        this._flushBufferAndStart();
      }
      return;
    }

    this._scheduleChunk(pcmInt16Data);
  }

  _flushBufferAndStart() {
    this._playbackStarted = true;
    this.nextPlayTime = this.audioContext.currentTime + 0.02;
    for (const chunk of this._chunkBuffer) {
      this._scheduleChunk(chunk);
    }
    this._chunkBuffer = [];
  }

  _scheduleChunk(pcmInt16Data) {
    // Convert Int16 → Float32
    const int16 = new Int16Array(pcmInt16Data);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      float32[i] = int16[i] / 32768;
    }

    // Create AudioBuffer
    const buffer = this.audioContext.createBuffer(1, float32.length, this.sampleRate);
    buffer.getChannelData(0).set(float32);

    // Schedule for gapless playback
    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(this.audioContext.destination);

    // Ensure we don't schedule in the past
    const now = this.audioContext.currentTime;
    if (this.nextPlayTime < now) {
      this.nextPlayTime = now;
    }

    source.start(this.nextPlayTime);
    this._lastChunkDuration = buffer.duration;
    this.nextPlayTime += buffer.duration;

    this.scheduledSources.push(source);

    // Clear previous end timeout since we have more data
    if (this._endTimeout) {
      clearTimeout(this._endTimeout);
      this._endTimeout = null;
    }
  }

  /**
   * Signal end of stream. Lets remaining buffers finish, then fires onPlaybackEnd.
   */
  endStream() {
    // Flush any remaining buffered chunks (short responses with < minBufferChunks)
    if (!this._playbackStarted && this._chunkBuffer.length > 0) {
      this._flushBufferAndStart();
    }

    if (!this.audioContext) return;

    // Calculate how long until the last scheduled buffer finishes
    const remaining = Math.max(0, this.nextPlayTime - this.audioContext.currentTime);

    this._endTimeout = setTimeout(() => {
      this.isPlaying = false;
      this._cleanup();
      if (this.onPlaybackEnd) this.onPlaybackEnd();
    }, remaining * 1000 + 50); // small buffer for safety
  }

  /**
   * Immediately stop all playback (for interrupt).
   */
  interrupt() {
    if (!this.isPlaying) return;

    if (this._endTimeout) {
      clearTimeout(this._endTimeout);
      this._endTimeout = null;
    }

    // Stop all scheduled sources
    for (const source of this.scheduledSources) {
      try {
        source.stop();
      } catch (e) {
        // Already stopped or not started
      }
    }

    this.isPlaying = false;
    this._cleanup();
    if (this.onPlaybackEnd) this.onPlaybackEnd();
  }

  /**
   * Clean up resources without triggering onPlaybackEnd.
   */
  dispose() {
    if (this._endTimeout) {
      clearTimeout(this._endTimeout);
      this._endTimeout = null;
    }
    for (const source of this.scheduledSources) {
      try {
        source.stop();
      } catch (e) {
        // Ignore
      }
    }
    this.scheduledSources = [];
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close().catch(() => {});
    }
    this.audioContext = null;
    this.isPlaying = false;
    this.nextPlayTime = 0;
  }

  _cleanup() {
    this.scheduledSources = [];
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close().catch(() => {});
    }
    this.audioContext = null;
  }
}
