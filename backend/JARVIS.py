# v0.27 adds Sesame CSM text-to-speech integration
import queue
import threading
import time
import base64
import requests
from groq import Groq
import pyinotify
import sys
import os

# Configuration: set USE_LOCAL_WHISPER=true to use local model instead of Groq API
USE_LOCAL_WHISPER = os.environ.get("USE_LOCAL_WHISPER", "false").lower() == "true"

# Only import heavy dependencies if using local whisper
if USE_LOCAL_WHISPER:
    import torch
    from transformers import pipeline
    import librosa

# Initialize a queue to handle new files as they come in rather than force them to process all at once which may cause errors
file_queue = queue.Queue()

# Initialize a list to hold the messages
persistent_messages = []

# TTS setting for audio responses (controlled by frontend toggle)
audio_tts_enabled = True

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", "ENTER_API_KEY_HERE"))

# TTS Configuration (controlled by frontend toggle, not environment variable)
DEEPINFRA_TOKEN = os.environ.get("DEEPINFRA_TOKEN", "")
TTS_OUTPUT_DIR = os.environ.get("TTS_OUTPUT_DIR", "/app/audio-server/outputs")
TTS_MAX_CHARS = int(os.environ.get("TTS_MAX_CHARS", "1000"))  # Max characters to synthesize
TTS_VOICE = os.environ.get("TTS_VOICE", "")  # Custom voice name (create via /v1/voices/add)
# TTS Provider: "orpheus" (Groq), "chatterbox" (DeepInfra), or "csm" (DeepInfra)
TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "chatterbox")
# Orpheus voice options: autumn, tara, leah, jess, leo, dan, mia, zac
TTS_ORPHEUS_VOICE = os.environ.get("TTS_ORPHEUS_VOICE", "autumn")
# Orpheus speech speed (1.0 = normal, 1.5 = faster, 2.0 = very fast)
TTS_ORPHEUS_SPEED = float(os.environ.get("TTS_ORPHEUS_SPEED", "1.0"))

# Ensure TTS output directory exists
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)

# Orpheus TTS max input size
ORPHEUS_MAX_CHARS = 200

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
    """Generate speech using configured TTS provider"""
    # Truncate long text to avoid cutoff issues
    original_len = len(text)
    if original_len > TTS_MAX_CHARS:
        # Try to truncate at a sentence boundary
        truncated = text[:TTS_MAX_CHARS]
        last_period = truncated.rfind('.')
        last_question = truncated.rfind('?')
        last_exclaim = truncated.rfind('!')
        last_sentence = max(last_period, last_question, last_exclaim)
        if last_sentence > TTS_MAX_CHARS // 2:
            text = truncated[:last_sentence + 1]
        else:
            text = truncated + "..."
        print(f"TTS: Truncated text from {original_len} to {len(text)} chars", file=sys.stderr)

    # Use Groq Orpheus TTS
    if TTS_PROVIDER == "orpheus":
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

    # DeepInfra providers (chatterbox, csm)
    if not DEEPINFRA_TOKEN:
        print("Warning: DEEPINFRA_TOKEN not set, skipping TTS", file=sys.stderr)
        return None

    # Select TTS provider endpoint
    if TTS_PROVIDER == "csm":
        endpoint = "https://api.deepinfra.com/v1/inference/sesame/csm-1b"
        payload = {"text": text, "response_format": "wav"}
    else:  # chatterbox (default)
        endpoint = "https://api.deepinfra.com/v1/inference/ResembleAI/chatterbox-turbo"
        payload = {"text": text, "output_format": "wav"}
        if TTS_VOICE:
            payload["voice"] = TTS_VOICE

    try:
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {DEEPINFRA_TOKEN}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        # API returns JSON with base64-encoded audio in data URL format
        data = response.json()
        audio_data_url = data.get("audio", "")

        # Parse data URL: "data:audio/wav;base64,<base64_data>"
        if audio_data_url.startswith("data:"):
            # Extract base64 portion after the comma
            base64_data = audio_data_url.split(",", 1)[1]
            audio_bytes = base64.b64decode(base64_data)
            return audio_bytes
        else:
            print(f"TTS API returned unexpected audio format", file=sys.stderr)
            return None

    except requests.exceptions.RequestException as e:
        print(f"TTS API error: {e}", file=sys.stderr)
        return None
    except (ValueError, KeyError) as e:
        print(f"TTS response parsing error: {e}", file=sys.stderr)
        return None

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

# Function to handle chat with Groq based on transcribed text
def handle_chat_with_groq(transcribed_text, enable_tts=True):
    global persistent_messages

    #append the user's message to the persistent messages list
    persistent_messages.append({
        "role": "user",
        "content": transcribed_text,
    })

    # ensure the messages list doesn't exceed the max_messages limit
    max_messages = 60
    if len(persistent_messages) > max_messages:
        # Remove the oldest message to keep the list within the max_messages limit
        persistent_messages.pop(0)
        # If the first message is from the assistant, remove it as well to ensure the list starts with a user message
        if persistent_messages[0]["role"] == "assistant":
            persistent_messages.pop(0)
    
    system_prompt = os.environ.get(
        "JARVIS_SYSTEM_PROMPT",
        "You are my good friend and AI companion who loves to roast me. You also love salamanders."
    )
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }] + persistent_messages


    
    chat_completion = groq_client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
    )

    # extract the llm's response and append it to the messages list
    response = chat_completion.choices[0].message.content
    persistent_messages.append({
        "role": "assistant",
        "content": response,
    })

    if len(persistent_messages) > max_messages:
        # Remove the oldest message to keep the list within the max_messages limit
        persistent_messages.pop(0)
        # If the first message is from the assistant, remove it as well to ensure the list starts with a user message
        if persistent_messages[0]["role"] == "assistant":
            persistent_messages.pop(0)

    response = chat_completion.choices[0].message.content

    # Print model response immediately so frontend can display it
    print("Model:", response)
    sys.stdout.flush()

    # Generate TTS audio if enabled (controlled by frontend toggle)
    if enable_tts:
        def generate_tts():
            audio_result = text_to_speech(response)
            if audio_result:
                timestamp = int(time.time() * 1000)
                # Handle list of chunks (Orpheus) or single bytes (DeepInfra)
                if isinstance(audio_result, list):
                    for i, audio_bytes in enumerate(audio_result):
                        audio_filename = f"response_{timestamp}_{i}.wav"
                        audio_path = os.path.join(TTS_OUTPUT_DIR, audio_filename)
                        with open(audio_path, 'wb') as f:
                            f.write(audio_bytes)
                        print(f"Audio: {audio_filename}")
                        sys.stdout.flush()
                else:
                    audio_filename = f"response_{timestamp}.wav"
                    audio_path = os.path.join(TTS_OUTPUT_DIR, audio_filename)
                    with open(audio_path, 'wb') as f:
                        f.write(audio_result)
                    print(f"Audio: {audio_filename}")
                    sys.stdout.flush()

        # Run TTS in background thread so text appears immediately
        tts_thread = threading.Thread(target=generate_tts, daemon=True)
        tts_thread.start()


# Load local Whisper model only if configured to use it
local_whisper_pipe = None
if USE_LOCAL_WHISPER:
    print("Loading local Whisper model...", file=sys.stderr)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-medium"
    local_whisper_pipe = pipeline("automatic-speech-recognition", model=model_id, device=device, torch_dtype=torch_dtype)
    print(f"Local Whisper model loaded on {device}", file=sys.stderr)
else:
    print("Using Groq Whisper API for transcription", file=sys.stderr)

# Function to process audio files, now designed to be run in a worker thread
def process_audio_files():
    global audio_tts_enabled
    while True:
        # Get the next audio file path from the queue
        audio_file_path = file_queue.get()
        try:
            if USE_LOCAL_WHISPER:
                # Use local Whisper model
                audio_data, _ = librosa.load(audio_file_path, sr=16000)
                result = local_whisper_pipe(audio_data)
                transcribed_text = result["text"]
            else:
                # Use Groq Whisper API
                transcribed_text = transcribe_with_groq(audio_file_path)

            print("Transcribed Text:", transcribed_text)
            sys.stdout.flush()
            # Use the audio TTS setting (controlled by frontend toggle)
            handle_chat_with_groq(transcribed_text, enable_tts=audio_tts_enabled)
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
                # Text input with TTS enabled
                text_message = line[9:].strip()
                if text_message:
                    handle_chat_with_groq(text_message, enable_tts=True)
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

# Signal that we're ready to receive messages
print("READY")
sys.stdout.flush()

notifier.loop()
