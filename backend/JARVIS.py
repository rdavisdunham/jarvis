# v0.32 - Long-term memory (Mem0 + Qdrant)
import queue
import threading
import time
from groq import Groq
import pyinotify
import sys
import os
import memory
import eou_detector

# STT Configuration: 'groq' (API batch) or 'local' (streaming faster-whisper on GPU)
# Backwards compat: map USE_LOCAL_WHISPER=true → STT_PROVIDER=local
_legacy_local = os.environ.get("USE_LOCAL_WHISPER", "false").lower() == "true"
STT_PROVIDER = os.environ.get("STT_PROVIDER", "local" if _legacy_local else "groq")

# Initialize a queue to handle new files as they come in rather than force them to process all at once which may cause errors
file_queue = queue.Queue()


class ContextManager:
    """
    Token-aware context management with rolling summarization.

    Uses a hybrid sliding window + rolling summary approach:
    [System Prompt] + [Rolling Summary] + [Recent Messages]
       ~500 tokens      ~2000 tokens       up to ~62k tokens
    """

    # Token thresholds
    SUMMARIZATION_TRIGGER = 65000  # ~50% of 131k limit
    HARD_LIMIT = 80000             # ~61% of limit
    MAX_SINGLE_MESSAGE = 20000    # Truncate very long messages
    MIN_RECENT_MESSAGES = 10      # Keep at least this many verbatim
    MIN_SUMMARIZATION_INTERVAL = 30  # Seconds between summarizations
    MAX_SUMMARY_TOKENS = 4000     # Condense summary if it exceeds this

    def __init__(self, groq_client):
        self.groq_client = groq_client
        self.rolling_summary = ""
        self.recent_messages = []
        self.last_summarization_time = 0
        self._lock = threading.Lock()

    def _estimate_tokens(self, text):
        """Estimate token count using chars/4 heuristic."""
        return len(text) // 4

    def _estimate_message_tokens(self, message):
        """Estimate tokens for a single message including role overhead."""
        return self._estimate_tokens(message.get("content", "")) + 4  # ~4 tokens for role/formatting

    def _get_total_tokens(self, system_prompt):
        """Calculate total tokens for current context."""
        total = self._estimate_tokens(system_prompt)
        if self.rolling_summary:
            # Summary is embedded in a system message
            total += self._estimate_tokens(self.rolling_summary) + 50  # overhead for formatting
        for msg in self.recent_messages:
            total += self._estimate_message_tokens(msg)
        return total

    def add_message(self, role, content, system_prompt):
        """Add a message and trigger summarization if needed."""
        with self._lock:
            # Truncate very long messages
            if self._estimate_tokens(content) > self.MAX_SINGLE_MESSAGE:
                content = content[:self.MAX_SINGLE_MESSAGE * 4]  # chars = tokens * 4
                print(f"Warning: Truncated very long {role} message", file=sys.stderr)

            self.recent_messages.append({
                "role": role,
                "content": content,
            })

            # Check if summarization is needed
            total_tokens = self._get_total_tokens(system_prompt)
            time_since_last = time.time() - self.last_summarization_time

            if total_tokens > self.SUMMARIZATION_TRIGGER and time_since_last > self.MIN_SUMMARIZATION_INTERVAL:
                self._perform_summarization(system_prompt)
            elif total_tokens > self.HARD_LIMIT:
                # Emergency truncation if we're over hard limit
                self._emergency_truncate()

    def _perform_summarization(self, system_prompt):
        """Summarize oldest messages using a fast LLM."""
        if len(self.recent_messages) <= self.MIN_RECENT_MESSAGES:
            return

        # Calculate how many messages to summarize (oldest 35%)
        num_to_summarize = max(1, int(len(self.recent_messages) * 0.35))

        # Ensure we keep at least MIN_RECENT_MESSAGES
        num_to_summarize = min(num_to_summarize, len(self.recent_messages) - self.MIN_RECENT_MESSAGES)

        if num_to_summarize < 2:
            return

        # Ensure we don't break user-assistant pairs
        messages_to_summarize = self.recent_messages[:num_to_summarize]
        if messages_to_summarize[-1]["role"] == "user":
            # Don't leave a user message without its response
            num_to_summarize -= 1
            messages_to_summarize = self.recent_messages[:num_to_summarize]

        if num_to_summarize < 2:
            return

        # Build conversation text for summarization
        conversation_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in messages_to_summarize
        ])

        # Include existing summary for context
        existing_context = ""
        if self.rolling_summary:
            existing_context = f"Previous conversation summary:\n{self.rolling_summary}\n\n"

        summarization_prompt = f"""{existing_context}New conversation to summarize:
{conversation_text}

Create a concise summary that preserves:
- Key facts (names, dates, numbers, preferences mentioned)
- Decisions and conclusions reached
- User context (their situation, goals, problems)
- Any commitments or action items
- The general tone/relationship dynamic

Write in third person (e.g., "The user asked about...", "The assistant explained...").
If there's a previous summary, merge the new information with it.
Keep the summary under 500 words."""

        try:
            completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a conversation summarizer. Create concise, informative summaries that preserve key context."},
                    {"role": "user", "content": summarization_prompt}
                ],
                model="openai/gpt-oss-20b",
                max_tokens=1000,
            )

            new_summary = completion.choices[0].message.content

            # Update state
            self.rolling_summary = new_summary
            self.recent_messages = self.recent_messages[num_to_summarize:]
            self.last_summarization_time = time.time()

            summary_tokens = self._estimate_tokens(new_summary)
            print(f"Context summarized: {num_to_summarize} messages -> {summary_tokens} tokens", file=sys.stderr)

            # Check if summary itself is too large
            if summary_tokens > self.MAX_SUMMARY_TOKENS:
                self._condense_summary()

        except Exception as e:
            print(f"Summarization failed: {e}, falling back to emergency truncation", file=sys.stderr)
            self._emergency_truncate()

    def _condense_summary(self):
        """Condense an oversized summary."""
        try:
            completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Condense this conversation summary to half its length while keeping the most important facts."},
                    {"role": "user", "content": self.rolling_summary}
                ],
                model="openai/gpt-oss-20b",
                max_tokens=500,
            )
            self.rolling_summary = completion.choices[0].message.content
            print("Summary condensed due to size", file=sys.stderr)
        except Exception as e:
            print(f"Summary condensation failed: {e}", file=sys.stderr)
            # Truncate summary as fallback
            self.rolling_summary = self.rolling_summary[:self.MAX_SUMMARY_TOKENS * 4]

    def _emergency_truncate(self):
        """Emergency truncation when summarization fails or we're over hard limit."""
        # Keep only the most recent messages
        if len(self.recent_messages) > self.MIN_RECENT_MESSAGES:
            removed = len(self.recent_messages) - self.MIN_RECENT_MESSAGES
            self.recent_messages = self.recent_messages[-self.MIN_RECENT_MESSAGES:]
            print(f"Emergency truncation: removed {removed} oldest messages", file=sys.stderr)

    def build_messages_for_api(self, system_prompt, memory_context=""):
        """Build the messages array for the API call."""
        messages = []

        # System prompt with memory context and embedded summary
        parts = [system_prompt]
        if memory_context:
            parts.append(f"\n\n{memory_context}")
        if self.rolling_summary:
            parts.append(f"\n\n[Previous conversation summary]\n{self.rolling_summary}\n[End of summary - recent conversation follows]")
        combined_system = "".join(parts)

        messages.append({
            "role": "system",
            "content": combined_system,
        })

        # Add recent messages
        messages.extend(self.recent_messages)

        return messages

# TTS setting for audio responses (controlled by frontend toggle)
audio_tts_enabled = True

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", "ENTER_API_KEY_HERE"))

# Initialize context manager for token-aware conversation handling
context_manager = ContextManager(groq_client)

# TTS Configuration
TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "groq")  # "groq" or "kokoro"
TTS_OUTPUT_DIR = os.environ.get("TTS_OUTPUT_DIR", "/app/audio-server/outputs")
# Groq Orpheus settings
# Orpheus voice options: autumn, tara, leah, jess, leo, dan, mia, zac
TTS_ORPHEUS_VOICE = os.environ.get("TTS_ORPHEUS_VOICE", "autumn")
# Orpheus speech speed (1.0 = normal, 1.5 = faster, 2.0 = very fast)
TTS_ORPHEUS_SPEED = float(os.environ.get("TTS_ORPHEUS_SPEED", "1.0"))

# Ensure TTS output directory exists
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)

# Max chunk sizes per provider
ORPHEUS_MAX_CHARS = 200
KOKORO_MAX_CHARS = 500

# Lazy-import Kokoro module only when needed
_tts_kokoro = None
def _get_kokoro():
    global _tts_kokoro
    if _tts_kokoro is None:
        import tts_kokoro
        _tts_kokoro = tts_kokoro
    return _tts_kokoro

print(f"TTS provider: {TTS_PROVIDER}", file=sys.stderr)

def split_text_for_tts(text, max_chars=ORPHEUS_MAX_CHARS):
    """Split text into chunks under max_chars, breaking at natural boundaries"""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        # Find the best break point within max_chars
        chunk = remaining[:max_chars]

        # Try to break at sentence boundaries first
        last_period = chunk.rfind('. ')
        last_question = chunk.rfind('? ')
        last_exclaim = chunk.rfind('! ')
        last_sentence = max(last_period, last_question, last_exclaim)

        if last_sentence > max_chars // 2:
            # Break at sentence boundary (include the punctuation)
            break_point = last_sentence + 1
        else:
            # Try to break at comma or other punctuation
            last_comma = chunk.rfind(', ')
            last_semicolon = chunk.rfind('; ')
            last_colon = chunk.rfind(': ')
            last_punct = max(last_comma, last_semicolon, last_colon)

            if last_punct > max_chars // 2:
                break_point = last_punct + 1
            else:
                # Break at last space
                last_space = chunk.rfind(' ')
                if last_space > max_chars // 2:
                    break_point = last_space
                else:
                    # Hard break at max_chars
                    break_point = max_chars

        chunks.append(remaining[:break_point].strip())
        remaining = remaining[break_point:].strip()

    return chunks

def text_to_speech(text):
    """Generate speech using Groq Orpheus TTS"""
    try:
        # Split text into chunks under 200 chars (Orpheus limit)
        chunks = split_text_for_tts(text)
        if len(chunks) > 1:
            print(f"TTS: Split text into {len(chunks)} chunks", file=sys.stderr)

        wav_chunks = []
        for chunk in chunks:
            response = groq_client.audio.speech.create(
                model="canopylabs/orpheus-v1-english",
                voice=TTS_ORPHEUS_VOICE,
                response_format="wav",
                input=chunk,
                speed=TTS_ORPHEUS_SPEED
            )
            wav_chunks.append(response.read())

        # Return list of audio chunks for sequential playback
        return wav_chunks
    except Exception as e:
        print(f"Groq TTS error: {e}", file=sys.stderr)
        return None


def generate_tts_streaming(text, output_dir):
    """Generate and write audio chunks immediately as each completes.

    This function combines TTS generation and file writing so that each
    audio chunk is written to disk immediately after its API call completes,
    rather than waiting for all chunks to be generated first.
    """
    if TTS_PROVIDER == "kokoro":
        _generate_tts_kokoro(text, output_dir)
    else:
        _generate_tts_groq(text, output_dir)


def _generate_tts_groq(text, output_dir):
    """Generate TTS via Groq Orpheus API."""
    try:
        chunks = split_text_for_tts(text, max_chars=ORPHEUS_MAX_CHARS)
        if len(chunks) > 1:
            print(f"TTS: Split text into {len(chunks)} chunks", file=sys.stderr)

        timestamp = int(time.time() * 1000)

        for i, chunk in enumerate(chunks):
            response = groq_client.audio.speech.create(
                model="canopylabs/orpheus-v1-english",
                voice=TTS_ORPHEUS_VOICE,
                response_format="wav",
                input=chunk,
                speed=TTS_ORPHEUS_SPEED
            )
            audio_bytes = response.read()

            audio_filename = f"response_{timestamp}_{i}.wav"
            audio_path = os.path.join(output_dir, audio_filename)
            with open(audio_path, 'wb') as f:
                f.write(audio_bytes)
            print(f"Audio: {audio_filename}")
            sys.stdout.flush()
    except Exception as e:
        print(f"Groq TTS streaming error: {e}", file=sys.stderr)


def _generate_tts_kokoro(text, output_dir):
    """Generate TTS via local Kokoro-82M model.

    Kokoro handles sentence splitting internally and runs at 50-80x
    real-time on RTX 3080. Generates full audio, saves as WAV, delivers via SSE.
    """
    try:
        kokoro = _get_kokoro()
        timestamp = int(time.time() * 1000)

        wav_bytes = kokoro.generate_speech(text)
        if wav_bytes:
            audio_filename = f"response_{timestamp}_0.wav"
            audio_path = os.path.join(output_dir, audio_filename)
            with open(audio_path, 'wb') as f:
                f.write(wav_bytes)
            print(f"Audio: {audio_filename}")
            sys.stdout.flush()
    except Exception as e:
        print(f"Kokoro TTS error: {e}", file=sys.stderr)

# Function to transcribe audio using Groq's Whisper API
def transcribe_with_groq(audio_file_path):
    # Debug: check file size
    file_size = os.path.getsize(audio_file_path)
    print(f"Audio file size: {file_size} bytes", file=sys.stderr)

    if file_size < 1000:
        print(f"Warning: Audio file very small, may be empty/corrupted", file=sys.stderr)

    filename = os.path.basename(audio_file_path)
    with open(audio_file_path, "rb") as audio_file:
        transcription = groq_client.audio.transcriptions.create(
            file=(filename, audio_file.read()),
            model="whisper-large-v3-turbo",
            response_format="text",
            language="en",  # Specify language to improve accuracy
        )
    return transcription

# Sentence-ending punctuation patterns used to detect TTS dispatch boundaries
_SENTENCE_ENDINGS = ('. ', '? ', '! ', '.\n', '?\n', '!\n')

def _dispatch_tts_sentence(text, pipeline_start):
    """Fire-and-forget: run TTS for a single sentence chunk in a background thread."""
    def _run(t=text, ps=pipeline_start):
        generate_tts_streaming(t, TTS_OUTPUT_DIR)
    threading.Thread(target=_run, daemon=True).start()


# Function to handle chat with Groq based on transcribed text
def handle_chat_with_groq(transcribed_text, enable_tts=True):
    pipeline_start = time.time()

    system_prompt = os.environ.get(
        "JARVIS_SYSTEM_PROMPT",
        "You are my good friend and AI companion who loves to roast me. You also love salamanders."
    )

    # Search long-term memory for relevant context
    memory_context = memory.search_memories(transcribed_text)

    # Add user message to context manager (handles summarization if needed)
    context_manager.add_message("user", transcribed_text, system_prompt)

    # Build messages array with system prompt, memory context, and rolling summary
    messages = context_manager.build_messages_for_api(system_prompt, memory_context=memory_context)

    t_llm = time.time()
    stream = groq_client.chat.completions.create(
        messages=messages,
        # model="llama-3.3-70b-versatile",
        model="openai/gpt-oss-120b",
        reasoning_effort="medium",  # options: "low", "medium", "high"
        stream=True,
    )

    full_response = ""
    tts_buffer = ""
    first_token_logged = False

    # Signal frontend that model text is about to stream
    print("MODEL_STREAM_START")
    sys.stdout.flush()

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta is None:
            continue

        if not first_token_logged:
            print(f"[timing] LLM first token: {(time.time() - t_llm) * 1000:.0f}ms", file=sys.stderr)
            first_token_logged = True

        full_response += delta

        # Send each chunk to frontend for live rendering
        # Encode newlines since the stdout protocol is line-based
        encoded = delta.replace('\\', '\\\\').replace('\n', '\\n')
        print(f"MODEL_CHUNK:{encoded}")
        sys.stdout.flush()

        # For Groq TTS: dispatch complete sentences as they arrive so audio
        # starts before the LLM finishes. Kokoro handles splitting internally
        # and gets the full text once streaming is done.
        if enable_tts and TTS_PROVIDER != "kokoro":
            tts_buffer += delta
            # Find the last sentence boundary in the buffer
            last_end = -1
            for ending in _SENTENCE_ENDINGS:
                pos = tts_buffer.rfind(ending)
                if pos + len(ending) > last_end + 1:
                    last_end = pos + len(ending) - 1
            if last_end >= 0:
                sentence = tts_buffer[:last_end + 1].strip()
                tts_buffer = tts_buffer[last_end + 1:]
                if sentence:
                    _dispatch_tts_sentence(sentence, pipeline_start)

    # Signal frontend that streaming is complete
    print("MODEL_STREAM_END")
    sys.stdout.flush()

    llm_ms = (time.time() - t_llm) * 1000
    print(f"[timing] LLM full response: {llm_ms:.0f}ms", file=sys.stderr)

    # Add assistant response to context manager
    context_manager.add_message("assistant", full_response, system_prompt)

    response_ready_ms = (time.time() - pipeline_start) * 1000
    print(f"[timing] Text response ready: {response_ready_ms:.0f}ms total", file=sys.stderr)

    # Extract and store memories in background
    memory.add_memory_background(transcribed_text, full_response)

    # Generate TTS audio if enabled (controlled by frontend toggle)
    if enable_tts:
        if TTS_PROVIDER == "kokoro":
            # Kokoro: dispatch full response now that streaming is complete
            def run_tts_kokoro():
                t_tts = time.time()
                generate_tts_streaming(full_response, TTS_OUTPUT_DIR)
                tts_ms = (time.time() - t_tts) * 1000
                total_ms = (time.time() - pipeline_start) * 1000
                print(f"[timing] TTS generation: {tts_ms:.0f}ms | Pipeline total (with TTS): {total_ms:.0f}ms", file=sys.stderr)
            threading.Thread(target=run_tts_kokoro, daemon=True).start()
        else:
            # Groq TTS: flush any remaining text that didn't end on a sentence boundary
            remainder = tts_buffer.strip()
            if remainder:
                _dispatch_tts_sentence(remainder, pipeline_start)


print(f"STT provider: {STT_PROVIDER}", file=sys.stderr)
if STT_PROVIDER == "local":
    print("Local STT handled by stt_streaming.py (transcripts arrive via stdin TEXT_TTS:)", file=sys.stderr)
else:
    print("Using Groq Whisper API for transcription", file=sys.stderr)

# Pending transcript buffer for EOU gating
_pending_transcript = ""
_pending_timer = None

def _eou_timeout():
    """Force-process pending transcript after timeout."""
    global _pending_transcript, _pending_timer
    if _pending_transcript:
        text = _pending_transcript
        _pending_transcript = ""
        _pending_timer = None
        print("Transcribed Text:", text)
        sys.stdout.flush()
        handle_chat_with_groq(text, enable_tts=audio_tts_enabled)

# Function to process audio files, now designed to be run in a worker thread
def process_audio_files():
    global audio_tts_enabled, _pending_transcript, _pending_timer
    while True:
        # Get the next audio file path from the queue
        audio_file_path = file_queue.get()
        try:
            # Use Groq Whisper API (local STT is handled by stt_streaming.py)
            transcribed_text = transcribe_with_groq(audio_file_path)

            # EOU gating: check if user is done speaking
            full_text = (_pending_transcript + " " + transcribed_text).strip() if _pending_transcript else transcribed_text

            if eou_detector.is_end_of_utterance(full_text):
                # Cancel any pending timer
                if _pending_timer:
                    _pending_timer.cancel()
                    _pending_timer = None
                _pending_transcript = ""

                print("Transcribed Text:", full_text)
                sys.stdout.flush()
                handle_chat_with_groq(full_text, enable_tts=audio_tts_enabled)
            else:
                # Not done speaking — buffer and signal
                _pending_transcript = full_text
                print("EOU_WAITING")
                sys.stdout.flush()
                # Start 3s timeout to force processing
                if _pending_timer:
                    _pending_timer.cancel()
                _pending_timer = threading.Timer(3.0, _eou_timeout)
                _pending_timer.daemon = True
                _pending_timer.start()
        except Exception as e:
            # Handle exceptions that occur during file processing
            print(f"Error processing {audio_file_path}: {e}", file=sys.stderr)
        finally:
            # Ensure the task is marked as done even if an error occurs
            file_queue.task_done()

# Set up pyinotify to monitor the Recordings directory
class EventHandler(pyinotify.ProcessEvent):
    def process_IN_CLOSE_WRITE(self, event):
        if event.pathname.endswith('.wav'):
            # Add the file path to the queue instead of processing it directly
            file_queue.put(event.pathname)

    def process_IN_MOVED_TO(self, event):
        if event.pathname.endswith('.wav'):
            file_queue.put(event.pathname)

# Function to read text messages from stdin
def process_stdin():
    global audio_tts_enabled
    while True:
        try:
            line = sys.stdin.readline()
            if line.startswith('TEXT:'):
                # Text input - no TTS by default
                text_message = line[5:].strip()
                if text_message:
                    handle_chat_with_groq(text_message, enable_tts=False)
            elif line.startswith('TEXT_TTS:'):
                # Text input with TTS — respect the current toggle state
                text_message = line[9:].strip()
                if text_message:
                    handle_chat_with_groq(text_message, enable_tts=audio_tts_enabled)
            elif line.startswith('INTERRUPT'):
                # Cancel ongoing Kokoro generation
                if TTS_PROVIDER == "kokoro":
                    kokoro = _get_kokoro()
                    kokoro.cancel_generation()
                print("INTERRUPT_ACK")
                sys.stdout.flush()
            elif line.startswith('MEMORY_MODE:'):
                # Update memory mode: full, read-only, off
                mode = line[12:].strip().lower()
                memory.set_mode(mode)
            elif line.startswith('TTS_SETTING:'):
                # Update TTS setting for audio responses
                setting = line[12:].strip().lower()
                audio_tts_enabled = setting == 'on'
                print(f"TTS setting updated: {'on' if audio_tts_enabled else 'off'}", file=sys.stderr)
                sys.stderr.flush()
        except Exception as e:
            print(f"Error reading stdin: {e}", file=sys.stderr)
            sys.stderr.flush()

# Start the worker thread
worker_thread = threading.Thread(target=process_audio_files, daemon=True)
worker_thread.start()

# Start the stdin reader thread
stdin_thread = threading.Thread(target=process_stdin, daemon=True)
stdin_thread.start()

wm = pyinotify.WatchManager()
handler = EventHandler()
notifier = pyinotify.Notifier(wm, handler)
# wm.add_watch is now using pyinotify.IN_CLOSE_WRITE | pyinotify.IN_MOVED_TO instead of IN_CREATE because we want to process the file after it has been fully written
wdd = wm.add_watch('uploads', pyinotify.IN_CLOSE_WRITE | pyinotify.IN_MOVED_TO)

print("Monitoring 'uploads' directory for new files. Press CTRL+C to stop.", file=sys.stderr)

# Pre-load EOU model if enabled
eou_detector.preload()

# Eager Kokoro model preload when using Kokoro provider
if TTS_PROVIDER == "kokoro":
    def _preload_kokoro():
        kokoro = _get_kokoro()
        kokoro.load_model()
        print("KOKORO_READY")
        sys.stdout.flush()
    threading.Thread(target=_preload_kokoro, daemon=True).start()

# Signal that we're ready to receive messages
print("READY")
sys.stdout.flush()

notifier.loop()
