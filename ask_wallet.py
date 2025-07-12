import os
import traceback
import logging
import streamlit as st
import requests
from typing import List
from tempfile import NamedTemporaryFile

from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer

from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# === Config === #
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("HF_TOKEN")
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
assert HF_TOKEN or USE_LOCAL_LLM, "HF_TOKEN is required unless USE_LOCAL_LLM is True"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "wallet_vectors")
MODEL_ID = os.getenv("MODEL_ID", "meta-llama/llama-3-8b-instruct")
REMOTE_API_URL = os.getenv("REMOTE_API_URL", "https://router.huggingface.co/novita/v3/openai/chat/completions")
LOCAL_API_URL = os.getenv("LOCAL_API_URL", "https://router.huggingface.co/novita/v3/openai/chat/completions")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LOG_FILE = os.getenv("LOG_FILE", "chat_logs.txt")
DATA_PATH = os.getenv("DATA_PATH", "data/")

# === Setup logging === #
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s %(message)s')

# === Embedding === #
class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_tensor=False).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t, convert_to_tensor=False).tolist() for t in texts]

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)

@st.cache_resource
def get_embedder():
    return EmbeddingModel(EMBED_MODEL)

# === PDF Ingestion to Qdrant === #
def ingest_pdfs_to_qdrant(data_path=DATA_PATH):
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = splitter.split_documents(documents)

    embedder = get_embedder()

    qdrant_client = QdrantClient(url=QDRANT_URL)
    qdrant_client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    Qdrant.from_documents(
        documents=text_chunks,
        embedding=embedder,
        collection_name=QDRANT_COLLECTION_NAME,
        client=qdrant_client,
    )
    print(f"✅ Qdrant DB created with {len(text_chunks)} chunks.")

# === Vector Store === #
class VectorStore:
    def __init__(self, embedder: EmbeddingModel):
        self.client = QdrantClient(url=QDRANT_URL)
        self.store = Qdrant(
            client=self.client,
            collection_name=QDRANT_COLLECTION_NAME,
            embedding=embedder
        )

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        return self.store.similarity_search(query, k=k)

# === Prompt Factory === #
def build_prompt(context: str, question: str) -> str:
    template = PromptTemplate(
        template="""
You are a factual assistant. Use only the information from the context to answer the user's question.
- Be **detailed**, **structured**, and **clear** in your answer.
- **Do not hallucinate** or make up information.
- Start the answer directly. Avoid small talk.

Context:
{context}

Question:
{question}

Detailed Answer:
""",
        input_variables=["context", "question"]
    )
    return template.format(context=context, question=question)

# === LLM Client === #
class GenericLLMClient:
    def __init__(self, model_id: str, token: str = None, use_local: bool = False):
        self.model_id = model_id
        self.token = token
        self.use_local = use_local
        self.url = LOCAL_API_URL if use_local else REMOTE_API_URL

    def generate(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.token and not self.use_local:
            headers["Authorization"] = f"Bearer {self.token}"

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use only the provided context. Do not hallucinate."},
            {"role": "user", "content": prompt}
        ]

        payload = {
            "model": self.model_id,
            "stream": False,
            "messages": messages,
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 1024
        }

        response = requests.post(self.url, headers=headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"API Error {response.status_code}: {response.text}")

        return response.json().get("message") if self.use_local else response.json()["choices"][0]["message"]["content"]

# === UI Helpers === #
def format_source_documents(docs: List[Document]) -> str:
    return "\n\n".join([
        f"📄 **{doc.metadata.get('source', 'Unknown Source')}**\n{doc.page_content.strip()}"
        for doc in docs
    ])

def load_pdf_text(uploaded_file):
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        loader = PyPDFLoader(tmp_file.name)
        pages = loader.load()
        return "\n\n".join([page.page_content for page in pages])

# === Streamlit App === #
def main():
    st.set_page_config(page_title="AskWallet Chatbot", page_icon="💬")
    st.title(":brain: AskWallet - AI Assistant")

    st.sidebar.title("Settings")
    use_local = st.sidebar.checkbox("Use Local LLM", value=USE_LOCAL_LLM)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.button(":inbox_tray: Rebuild Vector DB from PDFs"):
        with st.spinner("Rebuilding vector DB..."):
            ingest_pdfs_to_qdrant()
            st.success("✅ Vector DB rebuilt successfully.")

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_prompt = st.chat_input("💬 Ask your question here...")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        try:
            with st.spinner("🔍 Searching and generating answer..."):
                embedder = get_embedder()
                vectorstore = VectorStore(embedder)
                retrieved_docs = vectorstore.retrieve(user_prompt)

                context_chunks = [doc.page_content for doc in retrieved_docs]
                context = "\n\n".join(context_chunks)
                prompt = build_prompt(context, user_prompt)

                llm = GenericLLMClient(MODEL_ID, HF_TOKEN, use_local=use_local)
                answer = llm.generate(prompt)
                sources = format_source_documents(retrieved_docs)

                response = f"🧠 **Answer:**\n\n{answer}\n\n---\n"
                st.chat_message("assistant").markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                logging.info(f"USER: {user_prompt}\nASSISTANT: {answer}\n")

        except Exception as e:
            st.error("❌ An error occurred.")
            st.exception(e)
            logging.error(f"Exception occurred: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
