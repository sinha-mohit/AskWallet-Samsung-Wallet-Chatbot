import os
import streamlit as st
import traceback
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer
import requests
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")
DB_FAISS_PATH = "vectorstore/db_faiss"
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # works with chat API
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # Use LLaMA 3 for higher quality responses (optional):
API_URL = "https://router.huggingface.co/novita/v3/openai/chat/completions"

# Custom embedding wrapper for LangChain compatibility
class LocalSentenceTransformer:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=False).tolist()

    def embed_documents(self, texts):
        return [self.model.encode(t, convert_to_tensor=False).tolist() for t in texts]

    def __call__(self, text):
        # This makes it compatible with LangChain's .similarity_search()
        return self.embed_query(text)

# Load FAISS vectorstore
def get_vectorstore():
    try:
        embedding_model = LocalSentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error("❌ Failed to load FAISS vector store.")
        st.exception(e)
        return None

# Custom prompt template for RAG
def set_custom_prompt():
    return PromptTemplate(
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


# Call HuggingFace Inference Client
def call_hf_chat(prompt):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_ID,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Use only the provided context. Do not hallucinate."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 1024
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]

# Format documents with metadata
def format_source_documents(docs):
    return "\n\n".join([
        f"📄 **{doc.metadata.get('source', 'Unknown Source')}**\n{doc.page_content.strip()}"
        for doc in docs
    ])

# Main Streamlit App
def main():
    st.set_page_config(page_title="AskWallet Chatbot", page_icon="💬")
    st.title("🧠 AskWallet - AI Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_prompt = st.chat_input("💬 Ask your question here...")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        try:
            with st.spinner("🔍 Searching and generating answer..."):
                vectorstore = get_vectorstore()
                if not vectorstore:
                    return

                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                docs = retriever.invoke(user_prompt)

                context = "\n\n".join([doc.page_content for doc in docs])
                prompt_template = set_custom_prompt()
                final_prompt = prompt_template.format(context=context, question=user_prompt)

                answer = call_hf_chat(final_prompt)
                source_texts = format_source_documents(docs)

                response_display = f"🧠 **Answer:**\n\n{answer}\n\n---\n**🔗 Source Documents:**\n{source_texts}"
                st.chat_message("assistant").markdown(response_display)
                st.session_state.messages.append({"role": "assistant", "content": response_display})

        except Exception as e:
            st.error("❌ An error occurred.")
            st.exception(e)
            print(traceback.format_exc())

if __name__ == "__main__":
    main()
