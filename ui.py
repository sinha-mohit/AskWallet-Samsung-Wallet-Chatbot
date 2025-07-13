import streamlit as st
from config import Settings
from logging_utils import log_section
from embedding import EmbeddingModel
from vectorstore import VectorStore, ingest_pdfs_to_qdrant
from llm_client.remote import RemoteHuggingFaceClient
from llm_client.local import LocalLLMClient
from llm_orchestrator import LLMOrchestrator

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


def main(log_file):
    settings = Settings()
    st.set_page_config(page_title="AskWallet Chatbot", page_icon="💬", layout="wide")
    st.title(":brain: AskWallet - AI Assistant")
    st.sidebar.title("Settings & Controls")
    use_local = st.sidebar.checkbox("Use Local LLM", value=settings.use_local_llm, key="use_local_llm_checkbox")
    if st.sidebar.button("Update Vector DB from PDFs"):
        st.session_state.confirm_rebuild = True
    if st.session_state.get("confirm_rebuild"):
        st.sidebar.warning("⚠️ This will add new documents to the DB. Recreate the collection for a full reset.")
        col1, col2 = st.sidebar.columns(2)
        if col1.button("Just Add New", key="add_new"):
            with st.spinner("Adding new documents to vector DB..."):
                ingest_pdfs_to_qdrant(force_recreate=False)
            st.session_state.confirm_rebuild = False
        if col2.button("Wipe & Rebuild", key="wipe_rebuild"):
            with st.spinner("Wiping and rebuilding vector DB..."):
                ingest_pdfs_to_qdrant(force_recreate=True)
            st.session_state.confirm_rebuild = False
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

                    llm_client = LocalLLMClient(settings.model_id, settings.local_api_url) if use_local else RemoteHuggingFaceClient(settings.model_id, settings.remote_api_url, settings.hf_token)
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
