# v0.1 is a simple script that transcribes a single audio file and sends the transcribed text to Groq for processing.
import torch
from transformers import pipeline
import librosa
from groq import Groq  # Make sure to import Groq

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

# Existing WhisperTEST.py content
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-medium"

pipe = pipeline(
    "automatic-speech-recognition",
    model=model_id,
    device=device,
    torch_dtype=torch_dtype,
)

audio_file_path = 'Recordings/test4.wav'
audio_data, _ = librosa.load(audio_file_path, sr=16000)

result = pipe(audio_data)
print("Transcribed Text:", result["text"])

# Call the function with the transcribed text
handle_chat_with_groq(result["text"])