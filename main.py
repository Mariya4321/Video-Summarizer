from flask import Flask, render_template, request
import torch
from yt_dlp import YoutubeDL
from transformers import pipeline
from pydub import AudioSegment
import os
from dotenv import load_dotenv
import librosa
import numpy as np

# Load environment variables from .env file
load_dotenv()

ydl_opts = {
    "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
    "outtmpl": "downloads/%(title)s.%(ext)s",
    "quiet": False,
    "no_warnings": False,
    "extract_audio": True,
    "audio_format": "mp3",
    "audio_quality": "192",
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "libmp3lame",
            "preferredquality": "192",
        }
    ],
    "socket_timeout": 30,
    "http_headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    },
    "keepvideo": False,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
# Allow override via .env file
device = os.getenv('DEVICE', device if device == 'cuda' else 'cpu')
if device == 'auto':
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Use Hugging Face Transformers ASR pipeline instead of huggingsound
asr_model = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=0 if device == 'cuda' else -1)
summarizer = pipeline("summarization")


app = Flask(__name__)


def transcript_video(file_name):
    audio = AudioSegment.from_file(file_name)
    chunk_size_ms = int(os.getenv('CHUNK_SIZE_MS', 180000))  # 3 minutes default
    chunks = [audio[i:i + chunk_size_ms] for i in range(0, len(audio), chunk_size_ms)]
    return chunks


def split_text_dynamically(text, max_tokens):
    """
    Dynamically splits text into chunks based on the token limit.
    The chunk size adjusts to fit the summarizer's token limit.
    """
    words = text.split()
    chunks = []
    chunk = []

    for word in words:
        chunk.append(word)

        if len(' '.join(chunk).split()) >= max_tokens:
            chunks.append(' '.join(chunk))
            chunk = []

    if chunk:
        chunks.append(' '.join(chunk))

    return chunks


def summarize_text(chunks):
    summary = []
    max_length = int(os.getenv('MAX_SUMMARY_LENGTH', 150))
    min_length = int(os.getenv('MIN_SUMMARY_LENGTH', 50))
    
    for j, short_sent in enumerate(chunks, 1):
        summary = summarizer(short_sent, max_length=max_length, min_length=min_length, do_sample=False)
        # Format as bullet points
    print("Summary:")
    return summary


@app.route('/')
def index():

    return render_template("index.html")


@app.route("/summery", methods=['GET', 'POST'])
def summery():
    url = ""
    final_text = []
    if request.method == 'POST':
        data = request.form
        url = data.get("videoUrl", "").strip()
        print(url)

    if url:
        try:
            # Create downloads directory if it doesn't exist
            os.makedirs("downloads", exist_ok=True)
            
            with YoutubeDL(ydl_opts) as ydl:
                print(f"Downloading: {url}")
                info_dict = ydl.extract_info(url, download=True)
                video_title = info_dict.get('title', 'video')
                print(f"Downloaded title: {video_title}")
            
            # Find the actual downloaded file
            downloads_dir = "downloads"
            actual_file = None
            
            # List files in downloads directory
            if os.path.exists(downloads_dir):
                all_files = os.listdir(downloads_dir)
                print(f"Files in downloads dir: {all_files}")
                
                # Look for MP3 files first
                mp3_files = [f for f in all_files if f.endswith('.mp3')]
                if mp3_files:
                    actual_file = os.path.join(downloads_dir, mp3_files[0])
                    print(f"Found MP3 file: {actual_file}")
                else:
                    # Look for any audio file
                    audio_files = [f for f in all_files if f.endswith(('.m4a', '.wav', '.webm', '.opus', '.flac', '.aac'))]
                    if audio_files:
                        actual_file = os.path.join(downloads_dir, audio_files[0])
                        print(f"Found audio file: {actual_file}")
            
            if not actual_file or not os.path.exists(actual_file):
                raise FileNotFoundError(f"No audio file found after download. Files in directory: {os.listdir(downloads_dir) if os.path.exists(downloads_dir) else 'N/A'}")
            
            print(f"Processing file: {actual_file}")
            chunks = transcript_video(actual_file)
            for i, chunk in enumerate(chunks, 1):
                chunk_path = f"chunk_{i}.mp3"
                chunk.export(chunk_path, format="mp3")
                
                # Load audio and transcribe using ASR pipeline
                audio_data, sr = librosa.load(chunk_path, sr=16000)
                result = asr_model(audio_data)
                text = result["text"]

                chunk_ = split_text_dynamically(text, 150)
                summary = summarize_text(chunk_)
                for idx, sentence in enumerate(summary[0]['summary_text'].split('. ')):
                    final_text.append(sentence)
        except Exception as e:
            print(f"Error: {e}")
            final_text = [f"Error processing video: {str(e)}"]

        return render_template("summery.html", text=final_text)
    
    # For GET requests without URL, return index page
    return render_template("index.html")


if __name__ == "__main__":
    debug_mode = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    port = int(os.getenv('PORT', 3000))
    
    print(f"🚀 Starting Flask app on port {port}...")
    print(f"📍 http://localhost:{port}")
    print(f"🔧 Debug mode: {debug_mode}")
    print(f"🖥️  Device: {device}")
    
    app.run(debug=debug_mode, port=port)
