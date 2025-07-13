import os
import traceback
import logging
import streamlit as st
import requests
from typing import List
from tempfile import NamedTemporaryFile
from abc import ABC, abstractmethod
import datetime

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from sentence_transformers import SentenceTransformer

from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, CollectionStatus

# === 0. Imports and Initial Setup === #
SYSTEM_PROMT = """
You are a intelligent AI software assistant. 
- Use ONLY the information from the CONTEXT above to answer the QUESTION.
- Do not hallucinate or use outside knowledge to answer.
- Start the answer directly. Avoid small talk or greetings.
- Use markdown formatting for clarity.
- Before answering, ensure you have understood the CONTEXT and QUESTION.
- Provide a concise, accurate, and well-structured answer.
- Understand that the CONTEXT may contain multiple documents.
- If the CONTEXT is too long, focus on the most relevant parts.
- Try to undertand the CONTEXT and QUESTION before answering.
- Try to understand the user's intent from the QUESTION.
- If QUESTION is not answerable with the given CONTEXT, respond with "I don't know" or "Not enough information".
"""
# === 1. Settings Management (Production-Grade) === #
# Load .env file first
load_dotenv()

class Settings(BaseSettings):
    """Manages application settings and secrets using Pydantic for validation."""

    # LLM and API Configuration
    use_local_llm: bool = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    hf_token: str = os.getenv("HF_TOKEN", "")
    model_id: str = os.getenv("MODEL_ID", "meta-llama/llama-3-8b-instruct")
    remote_api_url: str = os.getenv("REMOTE_API_URL", "https://router.huggingface.co/novita/v3/openai/chat/completions")
    local_api_url: str = os.getenv("LOCAL_API_URL", "http://localhost:11434/api/chat")

    # Qdrant Configuration
    qdrant_url: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "wallet_vectors")

    # Embedding Model
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embed_dimension: int = 384  # Default for MiniLM-L6-v2 (override manually if needed)

    # Local Data and Logging
    log_file: str = os.getenv("LOG_FILE", "chat_logs.txt")
    data_path: str = os.getenv("DATA_PATH", "data/")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Instantiate settings
settings = Settings()
assert settings.hf_token or settings.use_local_llm, "HF_TOKEN is required unless USE_LOCAL_LLM is True"

# === 2. Setup Logging === #
def get_log_filename():
    """Generate a unique log filename with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"chat_logs_{timestamp}.txt"

settings.log_file = get_log_filename()
logging.basicConfig(
    filename=settings.log_file,
    level=logging.INFO,
    format='\n%(asctime)s | %(levelname)s | %(message)s\n'  # Newline before each log entry for readability
)


def log_section(tag: str, message: str):
    logging.info(f"\n{'='*20} [{tag}] {'='*20}\n{message}\n{'='*50}\n")


# === 3. Embedding Model Wrapper === #
class EmbeddingModel(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_tensor=False).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()


# === 4. LLM Client Abstraction (Modular and Maintainable) === #
class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    @abstractmethod
    def generate(self, messages: List[dict]) -> str:
        pass

class RemoteHuggingFaceClient(LLMClient):
    """Client for remote Hugging Face Inference API."""
    def __init__(self, model_id: str, api_url: str, token: str):
        self.model_id = model_id
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def generate(self, messages: List[dict]) -> str:
        payload = {
            "model": self.model_id, "stream": False, "max_tokens": 1024, "temperature": 0.3, "top_p": 0.9,
            "messages": messages,
            "stop": None
        }
        logging.info(f"Sending payload to REMOTE LLM API: {payload}")
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            logging.info(f"API response: {response.json()}")
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            raise RuntimeError("Request to the language model timed out.")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")

class LocalLLMClient(LLMClient):
    """Client for a local llama.cpp model via HTTP API."""
    def __init__(self, model_id: str, api_url: str):
        self.model_id = model_id
        self.api_url = api_url  # Example: http://localhost:8000/v1/completions
        self.headers = {"Content-Type": "application/json"}

    def generate(self, messages: List[dict]) -> str:
        # Build prompt from chat history
        system_prompt = next((m["content"] for m in messages if m["role"] == "system"), SYSTEM_PROMT.strip())
        user_and_assistant = "\n\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages if m["role"] != "system"])
        full_prompt = f"{system_prompt}\n\n{user_and_assistant}"

        payload = {
            "model": self.model_id,
            "prompt": full_prompt,
            "max_tokens": 1024,
            "temperature": 0.3,
            "top_p": 0.9,
            "stop": None,
            "stream": False
        }

        logging.info(f"Sending payload to LOCAL LLM API: {payload}")
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()["content"]
        except requests.exceptions.Timeout:
            raise RuntimeError("Request to the local language model timed out.")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Local API request failed: {e}")



# === 5. Vector Store and Ingestion === #
class VectorStore:
    def __init__(self, embedder: Embeddings):
        self.client = QdrantClient(url=settings.qdrant_url)
        self.store = Qdrant(
            client=self.client,
            collection_name=settings.qdrant_collection_name,
            embeddings=embedder,
        )

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        return self.store.similarity_search(query, k=k)

def ingest_pdfs_to_qdrant(force_recreate: bool = False):
    """Loads PDFs, splits them, and upserts them into Qdrant."""
    client = QdrantClient(url=settings.qdrant_url)
    embedder = get_embedder()

    # Check if collection exists
    try:
        collection_info = client.get_collection(collection_name=settings.qdrant_collection_name)
        if collection_info.status != CollectionStatus.GREEN:
            force_recreate = True # Recreate if collection is unhealthy
    except Exception: # Catches connection errors or 404 if collection doesn't exist
        force_recreate = True

    if force_recreate:
        st.sidebar.warning("Recreating Qdrant collection...")
        client.recreate_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=VectorParams(size=settings.embed_dimension, distance=Distance.COSINE),
        )
        st.sidebar.success("Collection recreated.")

    st.sidebar.info("Loading documents...")
    loader = DirectoryLoader(settings.data_path, glob='*.pdf', loader_cls=PyPDFLoader, show_progress=True)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = splitter.split_documents(documents)
    
    st.sidebar.info(f"Embedding and upserting {len(text_chunks)} chunks...")
    Qdrant.from_documents(
        documents=text_chunks,
        embedding=embedder,
        collection_name=settings.qdrant_collection_name,
        url=settings.qdrant_url,
    )
    st.sidebar.success(f"✅ Qdrant DB updated with {len(text_chunks)} chunks.")

# === 6. Caching and App Logic (Efficient and Responsive) === #

@st.cache_resource
def get_embedder():
    """Cached function to load the embedding model once."""
    return EmbeddingModel(settings.embed_model)

@st.cache_resource(show_spinner=False)
def get_vectorstore():
    """Cached function to create and reuse the VectorStore and its DB connection."""
    return VectorStore(get_embedder())

@st.cache_resource
def get_llm_client(use_local: bool) -> LLMClient:
    """Cached factory to get the appropriate LLM client."""
    if use_local:
        return LocalLLMClient(model_id=settings.model_id, api_url=settings.local_api_url)
    return RemoteHuggingFaceClient(model_id=settings.model_id, api_url=settings.remote_api_url, token=settings.hf_token)

def build_prompt(context: str, question: str) -> str:
    """Builds a robust prompt with clear instructions and context."""
    template = PromptTemplate(
    template="""
CONTEXT:
---------
{context}
---------
QUESTION:
{question}

Detailed Answer:
""",
        input_variables=["context", "question"]
    )
    return template.format(context=context, question=question)

def format_source_documents(docs: List[Document]) -> str:
    """Formats retrieved documents for display."""
    return "\n\n".join([
        f"📄 **Source: {doc.metadata.get('source', 'N/A')} (Page: {doc.metadata.get('page', 'N/A')})**\n{doc.page_content.strip()}"
        for doc in docs
    ])

MEMORY_WINDOW = 6  # Number of recent messages to keep in LLM payload
SUMMARY_TRIGGER = 12  # Summarize if history exceeds this

def summarize_history(messages: List[dict]) -> str:
    """Advanced summarization of older chat history using LLM."""
    history = [m for m in messages if m["role"] in ("user", "assistant")]
    if not history:
        return ""
    # Use a simple concatenation as fallback
    raw_summary = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])
    # Use LLM to summarize the raw history
    llm = get_llm_client(settings.use_local_llm)
    summary_prompt = [
        {"role": "system", "content": "You are a helpful assistant. Summarize the following conversation history for context retention. Be concise and preserve important details."},
        {"role": "user", "content": raw_summary}
    ]
    try:
        summary = llm.generate(summary_prompt)
        return f"Summary of previous conversation:\n{summary}"
    except Exception as e:
        logging.error(f"Summarization failed: {e}")
        return f"Summary of previous conversation:\n{raw_summary}"

# === 7. Streamlit App UI (with UX Improvements) === #
def main():
    st.set_page_config(page_title="AskWallet Chatbot", page_icon="💬", layout="wide")
    st.title(":brain: AskWallet - AI Assistant")

    # --- Sidebar for Settings and Controls ---
    st.sidebar.title("Settings & Controls")
    use_local = st.sidebar.checkbox("Use Local LLM", value=settings.use_local_llm, key="use_local_llm_checkbox")
    
    # Safer DB Rebuild with Confirmation
    if st.sidebar.button("Update Vector DB from PDFs"):
        st.session_state.confirm_rebuild = True

    if st.session_state.get("confirm_rebuild"):
        st.sidebar.warning("⚠️ This will add new documents to the DB. Recreate the collection for a full reset.")
        col1, col2 = st.sidebar.columns(2)
        if col1.button("Just Add New", key="add_new"):
            with st.spinner("Adding new documents to vector DB..."):
                ingest_pdfs_to_qdrant(force_recreate=False)
            st.session_state.confirm_rebuild = False # Reset state
        if col2.button("Wipe & Rebuild", key="wipe_rebuild"):
            with st.spinner("Wiping and rebuilding vector DB..."):
                ingest_pdfs_to_qdrant(force_recreate=True)
            st.session_state.confirm_rebuild = False # Reset state
            
    try:
        client = QdrantClient(url=settings.qdrant_url)
        client.get_collections()
        st.sidebar.success("🟢 Qdrant connected")
    except Exception:
        st.sidebar.error("🔴 Qdrant not reachable!")


    # --- Main Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMT.strip()}
        ]

    log_section("STARTUP", f"Session started. Log file: {settings.log_file}")

    for msg in st.session_state.messages[1:]:  # Skip system prompt for display
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_prompt := st.chat_input("💬 Ask your question here..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        log_section("USER INPUT", f"User prompt received:\n{user_prompt}")
        with st.chat_message("user"):
            st.markdown(user_prompt)

        try:
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching and generating answer..."):
                    # 1. Retrieve documents
                    vectorstore = get_vectorstore()
                    retrieved_docs = vectorstore.retrieve(user_prompt)
                    log_section("VECTOR SEARCH", f"Retrieved {len(retrieved_docs)} documents for query: {user_prompt}\nMetadata: {[doc.metadata for doc in retrieved_docs]}")

                    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    log_section("CONTEXT", f"Context for prompt:\n{context}")

                    # 2. Advanced context and memory management
                    messages = st.session_state.messages.copy()
                    summary_msg = None
                    if len(messages) > SUMMARY_TRIGGER:
                        summary = summarize_history(messages[:-MEMORY_WINDOW])
                        summary_msg = {"role": "system", "content": summary}
                        log_section("SUMMARY", f"History summarized:\n{summary}")
                    window_msgs = [m for m in messages if m["role"] == "system"]
                    window_msgs += [m for m in messages if m["role"] in ("user", "assistant")][-MEMORY_WINDOW:]
                    if summary_msg:
                        window_msgs.insert(1, summary_msg)
                    window_msgs.append({"role": "system", "content": f"CONTEXT:\n---------\n{context}\n---------"})
                    log_section("PAYLOAD", f"LLM payload messages:\n{window_msgs}")

                    # 3. Generate Answer
                    llm = get_llm_client(use_local)
                    answer = llm.generate(window_msgs)
                    log_section("LLM RESPONSE", f"Generated answer:\n{answer}")

                    # 4. Format and Display Response
                    sources = format_source_documents(retrieved_docs)
                    response = f"🧐 **Answer:**\n\n{answer}\n\n---\n"
                    st.markdown(response)

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    log_section("CHAT HISTORY", f"USER: {user_prompt}\nASSISTANT: {answer}\nFull history: {st.session_state.messages}")

        except Exception as e:
            log_section("ERROR", f"Exception occurred:\n{traceback.format_exc()}")
            st.error(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()