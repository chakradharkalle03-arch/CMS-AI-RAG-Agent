# Carbon Management System -- User Manual

## 1. Introduction

This system is designed to help users upload documents (e.g., Excel
FAQs, PDF manuals), convert them into searchable knowledge using
embeddings + Chroma vector DB, and then ask natural language questions
via a web interface (FastAPI/Streamlit UI).

## 2. System Requirements

-   Python 3.10+
-   Installed dependencies (`requirements.txt`)
-   Internet connection (for Google Gemini API)

## 3. Installation & Setup

Clone or copy the project to your machine.

Create and activate a virtual environment:

``` bash
conda create -n carbonrag python=3.10 -y
conda activate carbonrag
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

Set environment variables in `main.py`:

``` bash
GEMINI_API_KEY=your_google_api_key
```

## 4. Running the Application

Start the FastAPI backend:

``` bash
uvicorn main:app --reload
```

→ The service runs at: <http://127.0.0.1:8000>

Open the UI in your browser to interact with the system.

## 5. Features & Workflow

### (a) File Upload

-   Navigate to `/upload`
-   Upload Excel (FAQ sheet) or PDF documents
-   System automatically parses and embeds content into Chroma Vector DB

### (b) Ask a Question

-   Go to `/ask` endpoint (or the Streamlit UI)
-   Enter a natural language query
-   System retrieves the most relevant chunks from the database
-   Response is generated via Gemini LLM and shown in the UI

### (c) Excel Report Generation

-   Navigate to `/generate`
-   Creates a structured Excel output (e.g.,
    `generated_user_manual.xlsx`) with Q&A results
-   Download directly to your system

### (d) Health & Metrics Check

Health check:

``` bash
curl http://127.0.0.1:8000/health
```

→ Should return:

``` json
{"status":"ok"}
```

Metrics check (Prometheus):

``` bash
curl http://127.0.0.1:8000/metrics
```

→ Returns Prometheus-compatible text output for monitoring

## 6. System Architecture

-   **Frontend**: FastAPI templates / Streamlit UI
-   **Backend**: FastAPI server with endpoints `/upload`, `/ask`,
    `/health`, `/metrics`
-   **Vector DB**: Chroma for storing document embeddings
-   **LLM**: Google Gemini for answer generation

**Data Flow**:

    User uploads document → parsed → embeddings created
    Queries → vector DB retrieval → Gemini LLM → answer displayed

## 7. Troubleshooting

-   **Error: API Key missing** → Check `.env` file
-   **Database not loading** → Delete old `chroma_db_excel` folder and
    restart server
-   **Wrong language output** → Update Excel FAQ to English or enable
    translator integration

## 8. Future Enhancements

-   Add translation layer (auto EN ↔ CN)
-   Add Docker deployment
-   Support multi-file uploads
-   Add Redis caching for faster query responses
