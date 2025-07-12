"""
AskWallet Chatbot - Production Grade with Local Fallback & Logging

Features:
- Online/Offline LLM switch (OpenLLM client for production)
- User query + response logging
- PDF upload for custom context
- Persistent chat history
- Offline PDF ingestion to FAISS store
"""

import os
import traceback
import streamlit as st
import requests
import logging
from typing import List

from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === Config === #
load_dotenv(find_dotenv())
HF_TOKEN = os.getenv("HF_TOKEN")
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"

DB_FAISS_PATH = "vectorstore/db_faiss"
MODEL_ID = "meta-llama/llama-3-8b-instruct"
REMOTE_API_URL = "https://router.huggingface.co/novita/v3/openai/chat/completions"
LOCAL_API_URL = "http://localhost:11434/api/chat"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOG_FILE = "chat_logs.txt"
DATA_PATH = "data/"

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

# === PDF Ingestion to FAISS === #
def ingest_pdfs_to_faiss(data_path=DATA_PATH, db_path=DB_FAISS_PATH):
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = splitter.split_documents(documents)

    embedder = EmbeddingModel(EMBED_MODEL)
    db = FAISS.from_documents(text_chunks, embedder)
    db.save_local(db_path)
    print(f"✅ FAISS DB created with {len(text_chunks)} chunks.")

# === Vector Store === #
class VectorStore:
    def __init__(self, db_path: str, embedder: EmbeddingModel):
        self.db = FAISS.load_local(db_path, embedder, allow_dangerous_deserialization=True)

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        retriever = self.db.as_retriever(search_kwargs={"k": k})
        return retriever.invoke(query)

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
    def __init__(self, model_id: str, token: str = None):
        self.model_id = model_id
        self.token = token
        self.url = LOCAL_API_URL if USE_LOCAL_LLM else REMOTE_API_URL

    def generate(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.token and not USE_LOCAL_LLM:
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

        if USE_LOCAL_LLM:
            return response.json()["message"]
        return response.json()["choices"][0]["message"]["content"]

# === UI Helpers === #
def format_source_documents(docs: List[Document]) -> str:
    return "\n\n".join([
        f"\U0001F4C4 **{doc.metadata.get('source', 'Unknown Source')}**\n{doc.page_content.strip()}"
        for doc in docs
    ])

def load_pdf_text(uploaded_file):
    loader = PyPDFLoader(uploaded_file.name)
    pages = loader.load()
    return "\n\n".join([page.page_content for page in pages])

# === Streamlit App === #
def main():
    st.set_page_config(page_title="AskWallet Chatbot", page_icon="💬")
    st.title("🧠 AskWallet - AI Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Button to ingest/rebuild FAISS vectorstore
    if st.button("📥 Rebuild Vector DB from PDFs"):
        with st.spinner("Rebuilding vector DB..."):
            ingest_pdfs_to_faiss()
            st.success("✅ Vector DB rebuilt successfully.")

    uploaded_file = st.file_uploader("📄 Upload a PDF for custom context", type="pdf")

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_prompt = st.chat_input("💬 Ask your question here...")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        try:
            with st.spinner("🔍 Searching and generating answer..."):
                embedder = EmbeddingModel(EMBED_MODEL)
                vectorstore = VectorStore(DB_FAISS_PATH, embedder)
                retrieved_docs = vectorstore.retrieve(user_prompt)

                context_chunks = [doc.page_content for doc in retrieved_docs]

                if uploaded_file:
                    context_chunks.append(load_pdf_text(uploaded_file))

                context = "\n\n".join(context_chunks)
                prompt = build_prompt(context, user_prompt)

                llm = GenericLLMClient(MODEL_ID, HF_TOKEN)
                answer = llm.generate(prompt)
                sources = format_source_documents(retrieved_docs)

                # response = f"🧠 **Answer:**\n\n{answer}\n\n---\n**🔗 Source Documents:**\n{sources}"
                response = f"🧠 **Answer:**\n\n{answer}\n\n---\n"
                st.chat_message("assistant").markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Log interaction
                logging.info(f"USER: {user_prompt}\nASSISTANT: {answer}\n")

        except Exception as e:
            st.error("❌ An error occurred.")
            st.exception(e)
            print(traceback.format_exc())


if __name__ == "__main__":
    main()
