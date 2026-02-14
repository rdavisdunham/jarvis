"""
Kokoro-82M TTS provider.

Generates speech from text using Kokoro's pipeline, which internally
splits text into sentences and generates audio for each. Audio chunks
are concatenated and returned as WAV bytes. Cancellation is supported
via threading.Event checked between sentence chunks.
"""

import io
import os
import struct
import sys
import threading
import time

import numpy as np

SAMPLE_RATE = 24000
VOICE = os.environ.get("TTS_KOKORO_VOICE", "af_heart")
SPEED = float(os.environ.get("TTS_KOKORO_SPEED", "1.0"))
LANG_CODE = os.environ.get("TTS_KOKORO_LANG", "a")

# Module-level state
_pipeline = None
_model_ready = threading.Event()
_cancel_event = threading.Event()


def load_model():
    """Load Kokoro pipeline to GPU (one-time, ~2-3 GB VRAM).

    Safe to call multiple times — only loads once. Sets _model_ready
    when done so callers can wait on it.
    """
    global _pipeline
    if _pipeline is not None:
        return

    from kokoro import KPipeline

    print(f"Kokoro: Loading pipeline (lang={LANG_CODE}, voice={VOICE})...", file=sys.stderr)
    _pipeline = KPipeline(lang_code=LANG_CODE, device="cuda")
    print("Kokoro: Model loaded successfully", file=sys.stderr)
    _model_ready.set()


def cancel_generation():
    """Signal the generator to stop at the next sentence boundary."""
    _cancel_event.set()


def generate_speech(text):
    """Generate speech from text, returning WAV bytes.

    Kokoro's pipeline yields audio for each sentence. We concatenate
    all chunks and wrap in a WAV header. Cancellation is checked
    between sentence chunks.
    """
    _model_ready.wait()
    _cancel_event.clear()

    try:
        t0 = time.monotonic()
        all_pcm = []
        total_samples = 0

        for result in _pipeline(text, voice=VOICE, speed=SPEED):
            if _cancel_event.is_set():
                print("Kokoro: Generation cancelled", file=sys.stderr)
                break

            if result.audio is not None:
                # result.audio is a numpy float32 array, values in [-1, 1]
                all_pcm.append(result.audio)
                total_samples += len(result.audio)

        if not all_pcm:
            return None

        # Concatenate all sentence audio and convert to int16 PCM
        audio = np.concatenate(all_pcm)
        pcm_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()

        elapsed = time.monotonic() - t0
        audio_duration = total_samples / SAMPLE_RATE
        rtf = audio_duration / elapsed if elapsed > 0 else 0
        print(f"Kokoro: {audio_duration:.1f}s audio in {elapsed:.2f}s ({rtf:.1f}x real-time)",
              file=sys.stderr)

        return _pcm_to_wav(pcm_bytes)

    except Exception as e:
        print(f"Kokoro TTS error: {e}", file=sys.stderr)
        return None


def _pcm_to_wav(raw_pcm_bytes, sample_rate=SAMPLE_RATE):
    """Wrap raw int16 PCM bytes in a WAV header."""
    buf = io.BytesIO()
    num_channels = 1
    sample_width = 2  # 16-bit
    data_size = len(raw_pcm_bytes)
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))  # PCM format
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * num_channels * sample_width))
    buf.write(struct.pack("<H", num_channels * sample_width))
    buf.write(struct.pack("<H", sample_width * 8))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(raw_pcm_bytes)
    return buf.getvalue()
