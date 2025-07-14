from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, CollectionStatus
import streamlit as st
from config import Settings
from embedding import EmbeddingModel
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, UnstructuredFileLoader
from pathlib import Path

class VectorStore:
    def __init__(self, embedder):
        settings = Settings()
        self.client = QdrantClient(url=settings.qdrant_url)
        self.store = Qdrant(
            client=self.client,
            collection_name=settings.qdrant_collection_name,
            embeddings=embedder,
        )

    def retrieve(self, query: str, k: int = 3) -> list:
        return self.store.similarity_search(query, k=k)


def clear_qdrant(force_recreate: bool = False):
    settings = Settings()
    client = QdrantClient(url=settings.qdrant_url)
    try:
        collection_info = client.get_collection(collection_name=settings.qdrant_collection_name)
        if collection_info.status != CollectionStatus.GREEN:
            force_recreate = True
    except Exception:
        force_recreate = True
    if force_recreate:
        st.sidebar.warning("Clearing Qdrant collection...")
        client.recreate_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=VectorParams(size=settings.embed_dimension, distance=Distance.COSINE),
        )
        st.sidebar.success("QDrant collection cleared.")
    
    # also delete all files in data_path folder
    st.sidebar.info(f"Deleting all files in {settings.data_path}...")
    data_path = Path(settings.data_path)
    if data_path.exists() and data_path.is_dir():
        for file in data_path.iterdir():
            if file.is_file():
                file.unlink()
        st.sidebar.info(f"Deleted all files in {settings.data_path}")

def ingest_pdfs_to_qdrant(force_recreate: bool = False):
    settings = Settings()
    client = QdrantClient(url=settings.qdrant_url)
    embedder = EmbeddingModel(settings.embed_model)
    try:
        collection_info = client.get_collection(collection_name=settings.qdrant_collection_name)
        if collection_info.status != CollectionStatus.GREEN:
            force_recreate = True
    except Exception:
        force_recreate = True
    if force_recreate:
        st.sidebar.warning("Recreating Qdrant collection...")
        client.recreate_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=VectorParams(size=settings.embed_dimension, distance=Distance.COSINE),
        )
        st.sidebar.success("Collection recreated.")
    st.sidebar.info("Loading documents...")
    # loader = DirectoryLoader(settings.data_path, glob='*.pdf', loader_cls=PyPDFLoader, show_progress=True)
    # documents = loader.load()
    # st.sidebar.info(f"Loaded {len(documents)} documents from {settings.data_path}")

    extensions_to_loader = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".py": TextLoader,
        ".cpp": TextLoader,
        ".c": TextLoader,
        ".java": TextLoader,
        ".json": TextLoader,
        ".csv": TextLoader,
        ".docx": UnstructuredFileLoader,
    }

    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    # Load all files in data_path recursively
    st.sidebar.info(f"Scanning {settings.data_path} for documents...")
    for file_path in Path(settings.data_path).rglob("*"):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if ext in extensions_to_loader:
                loader_cls = extensions_to_loader[ext]
                try:
                    loader = loader_cls(str(file_path))
                    docs = loader.load()
                    if not docs:
                        st.sidebar.warning(f"No content found in {file_path.name}")
                        continue

                    st.sidebar.info(f"Loaded {len(docs)} documents from {file_path.name}")
                    # for each doc in documents, save only page_content
                    documents_contents = [Document(page_content=doc.page_content) for doc in docs]

                    # Split documents into chunks
                    text_chunks = splitter.split_documents(documents_contents)
                    print(f"Total text chunks: {len(text_chunks)}")  # Debug: print number of text chunks
                    st.sidebar.info(f"Embedding and upserting {len(text_chunks)} chunks...")

                    Qdrant.from_documents(
                        documents=text_chunks,
                        embedding=embedder,
                        collection_name=settings.qdrant_collection_name,
                        url=settings.qdrant_url,
                    )

                    # documents.extend(docs)
                    st.sidebar.info(f"Loaded {len(docs)} documents from {file_path.name}")
                except Exception as e:
                    st.sidebar.error(f"Failed to load {file_path.name}: {e}")    

    st.sidebar.success(f"✅ Qdrant DB updated with {len(text_chunks)} chunks.")
