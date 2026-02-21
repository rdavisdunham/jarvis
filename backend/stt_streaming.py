"""
Streaming STT server using faster-whisper + whisper_streaming.

WebSocket server on port 9001 (internal). Accepts binary PCM chunks (16-bit, 16kHz mono)
from the Node.js relay, feeds them to whisper_streaming's OnlineASRProcessor for
intelligent partial results with local agreement policy.

Protocol:
  Browser → Node.js → here:
    - Binary messages: raw Int16 PCM audio chunks (~100ms each)
    - JSON: { "type": "finalize" } → flush buffer and emit final transcript
    - JSON: { "type": "cancel" } → discard current buffer

  Here → Node.js → Browser:
    - JSON: { "type": "partial", "text": "..." }
    - JSON: { "type": "final", "text": "..." }
"""

import asyncio
import json
import logging
import os
import sys
import numpy as np
import eou_detector

log = logging.getLogger('stt_streaming')
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter('[STT] %(message)s'))
    log.addHandler(_h)
    log.propagate = False

# Configuration
WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'medium.en')
DEVICE = 'cuda' if os.environ.get('STT_DEVICE', 'cuda') == 'cuda' else 'cpu'
COMPUTE_TYPE = os.environ.get('STT_COMPUTE_TYPE', 'float16')
PORT = int(os.environ.get('STT_PORT', '9001'))
EOU_TIMEOUT = float(os.environ.get('EOU_TIMEOUT', '3.0'))

# Lazy-loaded model
_model = None

def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        log.info(f'Loading faster-whisper model: {WHISPER_MODEL} on {DEVICE} ({COMPUTE_TYPE})')
        _model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
        log.info('Model loaded')
    return _model


class StreamingTranscriber:
    """Accumulates PCM chunks and runs transcription on finalize."""

    def __init__(self):
        self._chunks = []
        self._sample_rate = 16000
        self._total_samples = 0

    def add_chunk(self, pcm_bytes):
        """Add a raw Int16 PCM chunk. Returns total duration in seconds."""
        self._chunks.append(pcm_bytes)
        self._total_samples += len(pcm_bytes) // 2  # int16 = 2 bytes per sample
        return self._total_samples / self._sample_rate

    def get_audio(self):
        """Convert accumulated chunks to float32 numpy array."""
        if not self._chunks:
            return None
        raw = b''.join(self._chunks)
        int16 = np.frombuffer(raw, dtype=np.int16)
        float32 = int16.astype(np.float32) / 32768.0
        return float32

    def duration(self):
        return self._total_samples / self._sample_rate

    def transcribe(self):
        """Run final transcription on accumulated audio. Returns text or None."""
        import time
        audio = self.get_audio()
        if audio is None or len(audio) < self._sample_rate * 0.3:  # < 300ms, skip
            return None

        t0 = time.monotonic()
        model = get_model()
        segments, info = model.transcribe(
            audio,
            beam_size=5,
            language='en',
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        text = ' '.join(seg.text.strip() for seg in segments).strip()
        elapsed = time.monotonic() - t0
        log.info(f'Final transcription: {elapsed:.2f}s for {len(audio)/self._sample_rate:.1f}s audio')
        return text if text else None

    def get_partial(self):
        """Quick partial transcription (beam=1, full audio)."""
        import time
        audio = self.get_audio()
        if audio is None or len(audio) < self._sample_rate * 0.3:
            return None

        t0 = time.monotonic()
        model = get_model()
        segments, _ = model.transcribe(
            audio,
            beam_size=1,
            language='en',
            vad_filter=False,
        )
        text = ' '.join(seg.text.strip() for seg in segments).strip()
        elapsed = time.monotonic() - t0
        log.info(f'Partial: {elapsed:.2f}s, text="{text[:80]}"')
        return text if text else None

    def reset(self):
        self._chunks = []
        self._total_samples = 0


async def handle_connection(websocket):
    """Handle a single WebSocket connection from the Node.js relay."""
    log.info('Client connected')
    transcriber = StreamingTranscriber()
    last_partial_text = ''
    last_eou = True  # Latest EOU result from partial loop (True = done speaking)
    partial_in_flight = False  # Simple flag — only one partial at a time
    streaming = False  # True while receiving audio chunks
    finalize_gen = 0  # Generation counter to invalidate stale force_finalize tasks

    async def partial_loop():
        """Continuously produce partials while audio is streaming."""
        nonlocal last_partial_text, partial_in_flight, last_eou
        while streaming:
            duration = transcriber.duration()
            if duration < 0.4:
                await asyncio.sleep(0.1)
                continue

            if partial_in_flight:
                await asyncio.sleep(0.05)
                continue

            partial_in_flight = True
            try:
                text = await asyncio.get_event_loop().run_in_executor(
                    None, transcriber.get_partial
                )
                if text and text != last_partial_text:
                    last_partial_text = text
                    # Run EOU check on the partial (~25ms on CPU)
                    eou_result = await asyncio.get_event_loop().run_in_executor(
                        None, eou_detector.is_end_of_utterance, text
                    )
                    last_eou = eou_result
                    await websocket.send(json.dumps({
                        'type': 'partial',
                        'text': text,
                        'eou': eou_result,
                    }))
            except Exception as e:
                log.warning(f'Partial transcription error: {e}')
            finally:
                partial_in_flight = False

            # Brief pause before next attempt
            await asyncio.sleep(0.1)

    partial_task = None

    chunk_count = 0
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # Binary PCM chunk
                transcriber.add_chunk(message)
                chunk_count += 1

                # Start partial loop on first chunk
                if not streaming:
                    streaming = True
                    chunk_count = 1
                    log.info('Audio streaming started, launching partial loop')
                    partial_task = asyncio.create_task(partial_loop())

                # Log every 10th chunk (~1s) so we can verify chunks are flowing
                if chunk_count % 10 == 0:
                    log.info(f'Receiving audio: {transcriber.duration():.1f}s ({chunk_count} chunks)')

            elif isinstance(message, str):
                # JSON control message
                try:
                    msg = json.loads(message)
                except json.JSONDecodeError:
                    continue

                if msg.get('type') == 'finalize':
                    # Stop partial loop and wait for in-flight to finish
                    streaming = False
                    finalize_gen += 1
                    if partial_task:
                        await partial_task
                        partial_task = None

                    # Use last partial if we have one (the whole point of streaming —
                    # the transcript is already done). Only run a full transcription
                    # if we never got a partial (very short utterance).
                    if last_partial_text:
                        text = last_partial_text
                        log.info(f'Using cached partial as final (0.00s): "{text}"')
                    else:
                        text = await asyncio.get_event_loop().run_in_executor(
                            None, transcriber.transcribe
                        )

                    if text:
                        # Use pre-computed EOU from partial loop if it says done.
                        # If not, re-run EOU on the complete transcript — the last
                        # partial may have run on an incomplete transcript and set
                        # last_eou=False even though the full utterance is now done.
                        final_eou = last_eou
                        if not last_eou:
                            log.info('last_eou=False — re-checking EOU on complete transcript')
                            final_eou = await asyncio.get_event_loop().run_in_executor(
                                None, eou_detector.is_end_of_utterance, text
                            )
                            log.info(f'Final EOU re-check: {"END" if final_eou else "CONTINUE"} | "{text[:60]}"')

                        if final_eou:
                            await websocket.send(json.dumps({
                                'type': 'final',
                                'text': text,
                            }))
                            log.info(f'Final (EOU=done): "{text}"')
                            transcriber.reset()
                            last_partial_text = ''
                            last_eou = True
                        else:
                            # Not done speaking — keep buffer, signal waiting
                            await websocket.send(json.dumps({
                                'type': 'eou_waiting',
                                'text': text,
                            }))
                            log.info(f'EOU waiting, buffered: "{text}"')
                            # If no new audio comes, force-send after 3s timeout
                            buffered_text = text
                            saved_gen = finalize_gen
                            async def force_finalize():
                                await asyncio.sleep(EOU_TIMEOUT)
                                nonlocal last_partial_text, last_eou
                                if finalize_gen != saved_gen:
                                    return
                                if transcriber.duration() > 0:
                                    try:
                                        await websocket.send(json.dumps({
                                            'type': 'final',
                                            'text': buffered_text,
                                        }))
                                        log.info(f'Force finalized (timeout): "{buffered_text}"')
                                    except Exception as e:
                                        log.warning(f'Force finalize error: {e}')
                                    transcriber.reset()
                                    last_partial_text = ''
                                    last_eou = True
                            asyncio.create_task(force_finalize())
                    else:
                        log.info('No speech detected in audio')
                        transcriber.reset()
                        last_partial_text = ''

                elif msg.get('type') == 'cancel':
                    streaming = False
                    if partial_task:
                        await partial_task
                        partial_task = None
                    transcriber.reset()
                    last_partial_text = ''
                    log.info('Cancelled')

    except Exception as e:
        log.info(f'Client disconnected: {e}')
    finally:
        streaming = False
        if partial_task and not partial_task.done():
            partial_task.cancel()
        log.info('Connection closed')


async def main():
    import websockets
    # Pre-load models
    get_model()
    eou_detector.preload()

    log.info(f'Starting STT WebSocket server on port {PORT}')
    print('STT_READY', flush=True)  # Signal to parent process

    async with websockets.serve(handle_connection, '0.0.0.0', PORT):
        await asyncio.Future()  # run forever


if __name__ == '__main__':
    asyncio.run(main())
