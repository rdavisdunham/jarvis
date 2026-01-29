# v0.26 adds option to use Groq's hosted Whisper model instead of local
import queue
import threading
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

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", "ENTER_API_KEY_HERE"))

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
def handle_chat_with_groq(transcribed_text):
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
    print("Model:", response)
    sys.stdout.flush()


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
            handle_chat_with_groq(transcribed_text)
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
    while True:
        try:
            line = sys.stdin.readline()
            if line.startswith('TEXT:'):
                text_message = line[5:].strip()
                if text_message:
                    handle_chat_with_groq(text_message)
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
