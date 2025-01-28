from flask import Flask, render_template, request
from huggingsound import SpeechRecognitionModel
import torch
from yt_dlp import YoutubeDL
from transformers import pipeline
from pydub import AudioSegment

ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": "%(title)s.%(ext)s",
    "postprocessors": [
        {  # Convert audio to .mp3
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }
    ],
}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english", device=device)
summarizer = pipeline("summarization")


app = Flask(__name__)


def transcript_video(file_name):
    audio = AudioSegment.from_file(file_name)
    chunk_size_ms = 3 * 60 * 1000  # 3 minutes
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
    for j, short_sent in enumerate(chunks, 1):
        summary = summarizer(short_sent, max_length=150, min_length=50, do_sample=False)
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
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            file_name = ydl.prepare_filename(info_dict).replace(".webm", ".mp3")
        print("File downloaded Successfully")
        chunks = transcript_video(file_name)
        try:
            for i, chunk in enumerate(chunks, 1):
                chunk.export(f"chunk_{i}.mp3", format="mp3")
                audio_paths = [f"chunk_{i}.mp3"]
                transcriptions = model.transcribe(audio_paths)
                text = transcriptions[0]["transcription"]

                chunk_ = split_text_dynamically(text, 150)
                summary = summarize_text(chunk_)
                for idx, sentence in enumerate(summary[0]['summary_text'].split('. ')):
                    final_text.append(sentence)
        except Exception as e:
            print(f"Error: {e}")

        return render_template("summery.html", text=final_text)


if __name__ == "__main__":
    app.run(debug=True, port=3000)
