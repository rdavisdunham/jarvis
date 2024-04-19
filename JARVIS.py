# v0.25 adds a queue to handle new files as they come in and changes the inotify watch to only trigger on file close so files aren't read before they are done uploading.
import queue
import threading
import torch
from transformers import pipeline
from groq import Groq
import librosa
import pyinotify
import sys

# Initialize a queue to handle new files as they come in rather than force them to process all at once which may cause errors
file_queue = queue.Queue()

# Function to handle chat with Groq based on transcribed text
def handle_chat_with_groq(transcribed_text):
    client = Groq(api_key="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")
    
    messages = [
        {
            "role": "system",
            "content": "You are my good friend and AI companion who loves to roast me. You also love salamanders.",
        },
        {
            "role": "user",
            "content": transcribed_text,
        }
    ]
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="mixtral-8x7b-32768",
    )
    
    response = chat_completion.choices[0].message.content
    print("Model:", response)
    sys.stdout.flush()


# Existing WhisperTEST.py content
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-medium"
pipe = pipeline("automatic-speech-recognition", model=model_id, device=device, torch_dtype=torch_dtype)

# Function to process audio files, now designed to be run in a worker thread
def process_audio_files():
    while True:
        # Get the next audio file path from the queue
        audio_file_path = file_queue.get()
        try:
            # Attempt to load and process the audio file
            audio_data, _ = librosa.load(audio_file_path, sr=16000)
            result = pipe(audio_data)
            print("Transcribed Text:", result["text"])
            handle_chat_with_groq(result["text"])
        except Exception as e:
            # Handle exceptions that occur during file processing
            print(f"Error processing {audio_file_path}: {e}")
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

# Start the worker thread
worker_thread = threading.Thread(target=process_audio_files, daemon=True)
worker_thread.start()

wm = pyinotify.WatchManager()
handler = EventHandler()
notifier = pyinotify.Notifier(wm, handler)
# wm.add_watch is now using pyinotify.IN_CLOSE_WRITE | pyinotify.IN_MOVED_TO instead of IN_CREATE because we want to process the file after it has been fully written
wdd = wm.add_watch('uploads', pyinotify.IN_CLOSE_WRITE | pyinotify.IN_MOVED_TO)

print("Monitoring 'Uploads' directory for new files. Press CTRL+C to stop.")
notifier.loop()