# 🎥 AI Video Summarizer

An intelligent web application that generates concise summaries from YouTube videos using AI-powered Natural Language Processing (NLP). Save time by extracting key insights from lengthy videos in seconds.

---

## 🚀 Features

✨ Generate AI-powered summaries from YouTube videos

📝 Extract and process video transcripts automatically

⚡ Fast and accurate content summarization

🎯 Highlights key points and important takeaways

🌐 Clean and user-friendly interface

📱 Responsive design for desktop and mobile devices

---

## 📸 Demo

### Home Page

<img width="1881" height="897" alt="Screenshot 2026-06-06 192421" src="https://github.com/user-attachments/assets/295a1e29-952c-42c4-a018-4442bf65738c" />
<img width="1878" height="822" alt="Screenshot 2026-06-06 192450" src="https://github.com/user-attachments/assets/a6101f22-bd7c-4d14-aca5-b3575abd65cd" />

### Generated Summary

*Add screenshot here*

---

## 🛠️ Tech Stack

### Frontend

* HTML5
* CSS3
* JavaScript

### Backend

* Python
* Flask

### AI / NLP

* OpenAI Model

### Tools

* VS code
* Git & GitHub

---

## 📂 Project Structure

```text
Video-Summarizer/
│
├── templates/
├── static/
├── main.py
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

# Video Summarizer - Setup Guide

## 📋 Project Files Created

### 1. **requirements.txt**

Contains all Python dependencies needed for the project:

- Flask 2.3.3 - Web framework
- torch 2.0.1 - PyTorch (GPU support)
- yt-dlp 2023.9.24 - YouTube video downloader
- transformers 4.32.0 - Hugging Face NLP models
- pydub 0.25.1 - Audio processing
- python-dotenv 1.0.0 - Environment variable management

### 2. **.env**

Environment configuration file with:

- Flask settings (FLASK_APP, FLASK_ENV, FLASK_DEBUG, PORT)
- Audio processing settings (CHUNK_SIZE_MS, MAX_SUMMARY_LENGTH, MIN_SUMMARY_LENGTH)
- Device configuration (DEVICE for cuda/cpu)

### 3. **.gitignore**

Protects sensitive files from being committed to git:

- .env file (never commit environment variables)
- Virtual environment folders
- Downloaded audio files (_.mp3, _.wav)
- Python cache files
- IDE settings

## 🚀 Installation Steps

### Step 1: Create Virtual Environment

```bash
python -m venv venv
```

### Step 2: Activate Virtual Environment

**Windows (PowerShell):**

```bash
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**

```bash
.\venv\Scripts\activate.bat
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Audio Processing Tools

**Windows:**

```bash
# Install FFmpeg using chocolatey
choco install ffmpeg
```

**macOS:**

```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**

```bash
sudo apt-get install ffmpeg
```

### Step 5: Update Main.py to Use .env

Add this to the top of your `main.py`:

```python
import os
from dotenv import load_dotenv

load_dotenv()

app.run(
    debug=os.getenv('FLASK_DEBUG', True),
    port=int(os.getenv('PORT', 3000))
)
```

## ⚠️ Known Issues

**huggingsound:** The package might have compatibility issues. If you encounter errors:

Option 1: Try installing without version specification

```bash
pip install huggingsound
```

Option 2: Use alternative speech recognition models:

```bash
pip install librosa
pip install librosa
```

Then update your main.py to use an alternative model.

## 📦 Package Versions

All packages have been tested with:

- Python 3.11.4
- Windows 10/11

## 🔍 Verify Installation

```bash
python -c "import Flask; import torch; import yt_dlp; print('✓ Core packages installed successfully')"
```

## 💡 Next Steps

1. Test the Flask app: `python main.py`
2. Access the app at: http://localhost:3000
3. Ensure FFmpeg is installed for audio processing
4. Configure your `.env` file as needed


## 🎯 How It Works

1. Enter a YouTube video URL.
2. The application fetches the transcript.
3. AI processes the transcript.
4. A concise summary is generated.
5. Users can quickly understand the video's main points without watching the entire content.

---

## 📌 Use Cases

* Students summarizing lectures
* Developers learning from tutorials
* Researchers reviewing conference talks
* Professionals extracting meeting insights
* Anyone who wants to save time on long videos

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature/new-feature
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push to GitHub

```bash
git push origin feature/new-feature
```

5. Open a Pull Request

---

## ⭐ Support

If you found this project helpful, please consider giving it a star ⭐ on GitHub.

---

## 👩‍💻 Author

**Mariya**

Java Full Stack Developer

GitHub: https://github.com/Mariya4321
