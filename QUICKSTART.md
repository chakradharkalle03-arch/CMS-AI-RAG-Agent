# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Set API Key
Set your Google Gemini API key in `main.py` or as environment variable:
```bash
export GOOGLE_GENAI_API_KEY=your_api_key_here
```

### Step 3: Run the Application
```bash
uvicorn main:app --reload
```

## 📝 Usage

1. **Upload Documents**: Go to `http://127.0.0.1:8000` and upload Excel or PDF files
2. **Ask Questions**: Use the web interface to ask questions about your documents
3. **View Results**: Get AI-powered answers based on your uploaded content

## 🔗 Key Endpoints

- **Home**: `http://127.0.0.1:8000/` - Main UI
- **Upload**: `http://127.0.0.1:8000/upload` - Upload documents
- **Health**: `http://127.0.0.1:8000/health` - Health check
- **Metrics**: `http://127.0.0.1:8000/metrics` - Prometheus metrics

## ⚡ Requirements

- Python 3.10+
- Google Gemini API Key
- Internet connection

For detailed documentation, see [README.md](README.md)
