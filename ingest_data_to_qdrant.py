import os
import argparse
from pathlib import Path
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, CollectionStatus

from config import Settings
from embedding import EmbeddingModel


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
        print("⚠️  Clearing Qdrant collection...")
        client.recreate_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=VectorParams(size=settings.embed_dimension, distance=Distance.COSINE),
        )
        print("✅ Qdrant collection cleared.")


def ingest_data_to_qdrant(force_recreate: bool = False):
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
        print("⚠️  Recreating Qdrant collection...")
        client.recreate_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=VectorParams(size=settings.embed_dimension, distance=Distance.COSINE),
        )
        print("✅ Qdrant collection recreated.")

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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )

    all_chunks = []
    print(f"📂 Scanning '{settings.data_path}' for supported files...")
    for file_path in Path(settings.data_path).rglob("*"):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if ext in extensions_to_loader:
                loader_cls = extensions_to_loader[ext]
                try:
                    loader = loader_cls(str(file_path))
                    docs = loader.load()
                    if not docs:
                        print(f"⚠️  No content in {file_path.name}")
                        continue

                    print(f"📄 Loaded {len(docs)} docs from {file_path.name}")
                    documents_contents = [Document(page_content=doc.page_content) for doc in docs]
                    text_chunks = splitter.split_documents(documents_contents)
                    print(f"🧩 {len(text_chunks)} chunks extracted from {file_path.name}")

                    print(f"🚀 Uploading {len(text_chunks)} chunks to Qdrant...")
                    Qdrant.from_documents(
                        documents=text_chunks,
                        embedding=embedder,
                        collection_name=settings.qdrant_collection_name,
                        url=settings.qdrant_url,
                    )

                    all_chunks.extend(text_chunks)
                except Exception as e:
                    print(f"❌ Failed to load {file_path.name}: {e}")

    if all_chunks:
        print(f"🚀 Uploading {len(all_chunks)} chunks to Qdrant...")
        print("✅ Qdrant DB updated successfully.")
    else:
        print("⚠️ No documents found to ingest.")


if __name__ == "__main__":
    print("\n📥 Qdrant Ingestion Utility")
    print("Choose an option:")
    print("1. Clear Qdrant collection only")
    print("2. Ingest documents to Qdrant")
    print("3. Clear and then ingest to Qdrant")

    choice = input("\nEnter 1, 2, or 3: ").strip()

    if choice == "1":
        clear_qdrant(force_recreate=True)
    elif choice == "2":
        ingest_data_to_qdrant(force_recreate=False)
    elif choice == "3":
        clear_qdrant(force_recreate=True)
        ingest_data_to_qdrant(force_recreate=False)
    else:
        print("❌ Invalid option. Exiting.")

    print("\n✅ Done.\n")