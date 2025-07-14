from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, CollectionStatus
import streamlit as st
from config import Settings
from embedding import EmbeddingModel
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
    loader = DirectoryLoader(settings.data_path, glob='*.pdf', loader_cls=PyPDFLoader, show_progress=True)
    documents = loader.load()
    st.sidebar.info(f"Loaded {len(documents)} documents from {settings.data_path}")

    # print(documents[0].page_content)  # Debug: print first document's content
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    
    # for each doc in documents, save only page_content
    documents_contents = [Document(page_content=doc.page_content) for doc in documents]
    text_chunks = splitter.split_documents(documents_contents)

    print(f"Total text chunks: {len(text_chunks)}")  # Debug: print number of text chunks
    print("Example chunk:", text_chunks[0])  # Debug: print first 100 chars of first chunk
    # print("chunk example:", text_chunks[0].page_content[:100])  # Debug: print first 100 chars of first chunk
    
    st.sidebar.info(f"Embedding and upserting {len(text_chunks)} chunks...")
    Qdrant.from_documents(
        documents=text_chunks,
        embedding=embedder,
        collection_name=settings.qdrant_collection_name,
        url=settings.qdrant_url,
    )
    st.sidebar.success(f"✅ Qdrant DB updated with {len(text_chunks)} chunks.")
