import streamlit as st
import os
import hashlib
from config import Settings
from logging_utils import log_section
from embedding import EmbeddingModel
from vectorstore import VectorStore, ingest_pdfs_to_qdrant, clear_qdrant
from dotenv import load_dotenv
from llm_client.remote import RemoteHuggingFaceClient
from llm_client.local import LocalLLMClient
from llm_orchestrator import LLMOrchestrator
# Update deprecated langchain imports
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

MEMORY_WINDOW = 3
SUMMARY_TRIGGER = 6
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
- If QUESTION is not answerable with the given CONTEXT, respond with "I don't know!" or "Not enough information!".
"""

# Utility to compute hash of uploaded file
def compute_file_hash(uploaded_file):
    hasher = hashlib.sha256()
    hasher.update(uploaded_file.getvalue())
    return hasher.hexdigest()


def main(log_file):
    # load environment variables
    load_dotenv()

    settings = Settings()

    st.set_page_config(page_title="AskWallet Chatbot", page_icon="💬", layout="wide")
    st.title(":brain: AskWallet - AI Assistant")
    st.sidebar.title("Settings & Controls")
    
    # Sidebar controls
    use_local = st.sidebar.checkbox("Use Local LLM", value=settings.use_local_llm, key="use_local_llm_checkbox")
    print("Using local LLM:", use_local)

    # Clear vector DB
    if st.sidebar.button("Clear Vector DB"):
        st.session_state.confirm_rebuild = True
        with st.spinner("Clearing all data from vector DB..."):
            clear_qdrant(force_recreate=True)
            # Reset uploaded file hashes
            st.session_state.uploaded_file_hashes = {}
            st.session_state.confirm_rebuild = False
        st.sidebar.success("Vector DB cleared!")

    # Upload and process PDF
    uploaded_file = st.sidebar.file_uploader("Upload PDF to Vector DB", type=["pdf"])

    if uploaded_file is not None:
        file_hash = compute_file_hash(uploaded_file)
        file_path = os.path.join(settings.data_path, uploaded_file.name)

        # Initialize file hash tracking if not already
        if "uploaded_file_hashes" not in st.session_state:
            st.session_state.uploaded_file_hashes = {}

        # Check if file is new or updated
        if (uploaded_file.name not in st.session_state.uploaded_file_hashes or
            st.session_state.uploaded_file_hashes[uploaded_file.name] != file_hash):

            with st.spinner("Uploading and adding data to vector DB..."):
                # Save file to disk
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Ingest into Qdrant
                ingest_pdfs_to_qdrant(force_recreate=False)

                # Update hash in session state
                st.session_state.uploaded_file_hashes[uploaded_file.name] = file_hash

            st.sidebar.success(f"✅ Added or updated {uploaded_file.name} in Vector DB!")
        else:
            st.sidebar.info("ℹ️ This file is already uploaded and unchanged.")

    # if st.sidebar.button("Update Vector DB from PDFs"):
    #     st.session_state.confirm_rebuild = True
    # if st.session_state.get("confirm_rebuild"):
    #     st.sidebar.warning("⚠️ This will add new documents to the DB. Recreate the collection for a full reset.")
    #     col1, col2 = st.sidebar.columns(2)
    #     if col1.button("Just Add New", key="add_new"):
    #         with st.spinner("Adding new documents to vector DB..."):
    #             ingest_pdfs_to_qdrant(force_recreate=False)
    #         st.session_state.confirm_rebuild = False
    #     if col2.button("Wipe & Rebuild", key="wipe_rebuild"):
    #         with st.spinner("Wiping and rebuilding vector DB..."):
    #             ingest_pdfs_to_qdrant(force_recreate=True)
    #         st.session_state.confirm_rebuild = False
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=settings.qdrant_url)
        client.get_collections()
        st.sidebar.success("🟢 Qdrant connected")
    except Exception:
        st.sidebar.error("🔴 Qdrant not reachable!")
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMT.strip()}
        ]
    log_section("STARTUP", f"Session started. Log file: {log_file}")
    for msg in st.session_state.messages[1:]:
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
                    vectorstore = VectorStore(EmbeddingModel(settings.embed_model))
                    retrieved_docs = vectorstore.retrieve(user_prompt)
                    log_section("VECTOR SEARCH", f"Retrieved {len(retrieved_docs)} documents for query: {user_prompt}\nMetadata: {[doc.metadata for doc in retrieved_docs]}")
                    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    log_section("CONTEXT", f"Context for prompt:\n{context}")
                    history = [m for m in st.session_state.messages if m["role"] in ("user", "assistant")]

                    llm_client = settings.get_llm_client(use_local)
                    orchestrator = LLMOrchestrator(settings, log_file, SYSTEM_PROMT, llm_client)
                    
                    payload_msg = orchestrator.build_payload(user_prompt, context, history)
                    log_section("PAYLOAD_MSG", f"LLM payload messages:\n{payload_msg}")
                    
                    answer = orchestrator.get_response(payload_msg)
                    log_section("LLM RESPONSE", f"Generated answer:\n{answer}")
                    
                    response = f"🧐 **Answer:**\n\n{answer}\n\n---\n"
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    log_section("CHAT HISTORY", f"USER: {user_prompt}\nASSISTANT: {answer}\nFull history: {st.session_state.messages}")
        except Exception as e:
            log_section("ERROR", f"Exception occurred:\n{e}")
            st.error(f"❌ An error occurred: {e}")
