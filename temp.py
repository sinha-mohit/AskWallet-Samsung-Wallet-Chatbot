import os
import streamlit as st
import traceback
from dotenv import load_dotenv, find_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from huggingface_hub import InferenceClient

# ✅ Load environment variables
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")
DB_FAISS_PATH = "vectorstore/db_faiss"
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # works with chat API
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # Use LLaMA 3 for higher quality responses (optional):

# ✅ Load FAISS vectorstore
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error("❌ Failed to load FAISS vector store.")
        st.exception(e)
        return None

# ✅ Custom prompt template for RAG
def set_custom_prompt():
    return PromptTemplate(
        template="""
You are an intelligent assistant. Use only the provided context to answer the question.
If the context does not contain the answer, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

# ✅ Call HuggingFace Inference Client
def call_hf_chat(prompt):
    client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# ✅ Format documents with metadata
def format_source_documents(docs):
    return "\n\n".join([
        f"📄 **{doc.metadata.get('source', 'Unknown Source')}**\n{doc.page_content.strip()}"
        for doc in docs
    ])

# ✅ Main Streamlit App
def main():
    st.set_page_config(page_title="Ask Wallet Chatbot", page_icon="💬")
    st.title("🧠 Ask Wallet AI Assistant")

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

                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                docs = retriever.get_relevant_documents(user_prompt)

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
