# RAG/baseline/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# RAG/baseline/
BASE_DIR = Path(__file__).resolve().parent
# RAG/
load_dotenv(BASE_DIR.parent / ".env")

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")

# --- LangSmith Tracing ---
LANGSMITH_TRACING_ENABLED = False  # True로 설정 시 LangSmith 추적 활성화
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "RAG-System-Final")

# --- Paths ---
FAISS_DB_PATH = BASE_DIR / "faiss_db" # RAG/baseline/faiss_db/

# --- Model Definitions ---
AVAILABLE_EMBEDDINGS = {
    "bge-m3": "BAAI/bge-m3",
    "e5e": "intfloat/e5-large-v2",
    "openai": "text-embedding-3-large"
}
AVAILABLE_RERANKERS = ["ko-rerank", "mini", "bge"]

# --- Default Settings ---
DEFAULT_EMBEDDING_ALIAS = "bge-m3"
DEFAULT_RERANKER_METHOD = "ko-rerank"
DEFAULT_RETRIEVER_K = 8
DEFAULT_RERANKER_TOP_K = 4
DEFAULT_LLM_MODEL_NAME = "gpt-4o"
DEFAULT_GPT_SCORING_MODEL = "gpt-4o"
DEFAULT_GPT_SUMMARIZATION_MODEL = "gpt-3.5-turbo"

# --- Upstage Fact checker ---
UPSTAGE_LABEL_TO_SCORE = {
    "grounded": 1.0,
    "notSure": 0.5,
    "notGrounded": 0.0
}