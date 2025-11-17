import os
import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Request, File, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
import uuid
import logging

from langdetect import detect
import jieba
import backoff

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import uvicorn

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# PDF processing imports
from PyPDF2 import PdfReader

# ======== Logging ========
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ======== Config ========
GOOGLE_API_KEY = os.environ.get("GOOGLE_GENAI_API_KEY") or "AIzaSyAkn6ZrT-SbcIaLFwtavhksfr4qTmX8twM"
UPLOAD_DIR = "uploads"
EXCEL_UPLOAD_DIR = os.path.join(UPLOAD_DIR, "excel")
PDF_UPLOAD_DIR = os.path.join(UPLOAD_DIR, "pdf")
PERSIST_DIRECTORY = "./chroma_db_excel"

# Create upload directories if they don't exist
os.makedirs(EXCEL_UPLOAD_DIR, exist_ok=True)
os.makedirs(PDF_UPLOAD_DIR, exist_ok=True)

# ======== Initialize Templates ========
templates = Jinja2Templates(directory="templates")

# ======== Initialize Embeddings and LLM ========
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0)

# Global variables
current_document_path = None
vectordb = None

# ======== Helper to safely close Chroma ========
def close_vectordb():
    """Close current Chroma instance to release file handles."""
    global vectordb
    if vectordb is not None:
        try:
            vectordb._client.reset()  # close sqlite + client
            logger.info("Closed previous Chroma DB connection.")
        except Exception as e:
            logger.warning(f"Error closing vectordb: {e}")
        vectordb = None

# ======== Process Excel ========
def process_excel_file(file_path):
    try:
        df = pd.read_excel(file_path)
        documents = []
        for index, row in df.iterrows():
            if pd.isna(row.iloc[0]) or (len(row) > 1 and pd.isna(row.iloc[1])):
                continue
            question = str(row.iloc[0]).strip()
            answer = str(row.iloc[1]).strip() if len(row) > 1 else ""
            if question and answer:
                text = f"Question: {question}\nAnswer: {answer}"
                metadata = {
                    "question": question,
                    "answer": answer,
                    "source": file_path,
                    "row_index": index,
                    "file_type": "excel",
                    "lang": detect(question) if question else "en"
                }
                documents.append(Document(page_content=text, metadata=metadata))
        return documents
    except Exception as e:
        logger.error(f"Error reading Excel file: {e}")
        return []

# ======== Process PDF ========
def process_pdf_file(file_path):
    try:
        documents = []
        pdf_reader = PdfReader(file_path)
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text and text.strip():
                lang = detect(text[:200]) if text else "en"
                if lang.startswith("zh"):
                    text = " ".join(jieba.cut(text))
                metadata = {
                    "source": file_path,
                    "page": page_num + 1,
                    "file_type": "pdf",
                    "lang": lang
                }
                documents.append(Document(page_content=text, metadata=metadata))
        return documents
    except Exception as e:
        logger.error(f"Error reading PDF file: {e}")
        return []

# ======== Build Vector Database ========
def build_vector_database(file_path, file_type):
    global current_document_path, vectordb

    # Close previous instance
    close_vectordb()

    # Remove old directory safely
    if os.path.exists(PERSIST_DIRECTORY):
        try:
            shutil.rmtree(PERSIST_DIRECTORY)
        except Exception as e:
            logger.warning(f"Error removing old DB folder: {e}")

    if file_type == "excel":
        documents = process_excel_file(file_path)
    elif file_type == "pdf":
        documents = process_pdf_file(file_path)
    else:
        raise ValueError("Unsupported file type")

    if not documents:
        raise ValueError("No documents were processed from the file.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    current_document_path = file_path
    logger.info(f"New Chroma DB built for: {file_path}")
    return vectordb

# ======== Prompt ========
prompt_template = """
You are a professional customer service assistant for a carbon management system.
Please answer strictly based on the provided context.
If no answer is found in the context, clearly say so and suggest contacting support.

Context:
{context}

Question:
{question}

Answer in the same language as the question.
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# ======== FastAPI App ========
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Metrics
REQUEST_COUNT = Counter("cms_requests_total", "Total requests", ["endpoint"])
REQUEST_LATENCY = Histogram("cms_request_latency_seconds", "Latency in seconds", ["endpoint"])

# ======== RAG Answer ========
def get_rag_answer(question: str):
    global vectordb
    if vectordb is None:
        return {"answer": "Please upload a document first.", "sources": [], "confidence": 0.0}
    try:
        q_lang = detect(question)
    except Exception:
        q_lang = "en"

    sources, contexts, scores = [], [], []
    try:
        results = vectordb.similarity_search_with_score(question, k=5)
        for doc, score in results:
            contexts.append(doc.page_content)
            scores.append(float(score))
            sources.append({
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "lang": doc.metadata.get("lang", q_lang),
                "score": float(score)
            })
    except Exception:
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(question)
        for doc in docs:
            contexts.append(doc.page_content)
            sources.append({
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "lang": doc.metadata.get("lang", q_lang),
                "score": None
            })

    context_text = "\n\n".join(contexts[:5])
    formatted_prompt = PROMPT.format(context=context_text, question=question)

    @backoff.on_exception(backoff.expo, Exception, max_time=30)
    def call_llm(prompt_text):
        resp = llm.invoke(prompt_text)
        return getattr(resp, "content", str(resp))

    try:
        answer_text = call_llm(formatted_prompt)
    except Exception as e:
        logger.error(f"LLM error: {e}")
        answer_text = "Error generating answer."

    avg_score = float(sum(scores) / len(scores)) if scores else None
    confidence = round((1.0 / (1.0 + avg_score)) * 100.0, 2) if avg_score else 0.0

    return {"answer": answer_text, "sources": sources, "confidence": confidence}

# ======== Routes ========
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "current_file": current_document_path})

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    global vectordb
    try:
        if file.filename.endswith(('.xlsx', '.xls')):
            file_type = "excel"
            upload_dir = EXCEL_UPLOAD_DIR
        elif file.filename.endswith('.pdf'):
            file_type = "pdf"
            upload_dir = PDF_UPLOAD_DIR
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        unique_filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
        file_path = os.path.join(upload_dir, unique_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        vectordb = build_vector_database(file_path, file_type)
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "message": f"File '{file.filename}' uploaded and DB built!", "current_file": file.filename}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.post("/ask", response_class=HTMLResponse)
async def ask(request: Request, question: str = Form(...)):
    REQUEST_COUNT.labels(endpoint="/ask").inc()
    with REQUEST_LATENCY.labels(endpoint="/ask").time():
        res = get_rag_answer(question)
    return templates.TemplateResponse("answer.html", {
        "request": request,
        "question": question,
        "answer": res["answer"],
        "sources": res["sources"],
        "confidence": res["confidence"]
    })

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return PlainTextResponse(data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

@app.get("/debug/db-content")
async def debug_db_content():
    global vectordb
    if vectordb is None:
        return {"error": "No vector database initialized"}
    retriever = vectordb.as_retriever(search_kwargs={"k": 50})
    docs = retriever.get_relevant_documents("")
    content = []
    for i, doc in enumerate(docs):
        content.append({
            "id": i + 1,
            "preview": doc.page_content[:80],
            "metadata": doc.metadata
        })
    return {"count": len(docs), "documents": content}

# ======== Main ========
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
