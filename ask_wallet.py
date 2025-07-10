import os
import streamlit as st
import traceback
from dotenv import load_dotenv, find_dotenv

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# ✅ Load environment variables
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # ✅ LLaMA 3

# ✅ Load FAISS vectorstore
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error("Failed to load FAISS vector store.")
        st.exception(e)
        print(traceback.format_exc())
        return None

# ✅ Create a custom prompt template
def set_custom_prompt():
    return PromptTemplate(
        template="""
Use the information from the context to answer the user's question.
Only use information from the context. If unsure, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

# ✅ Load LLaMA 3 from HuggingFace Endpoint
def load_llm(repo_id, hf_token):
    if not hf_token:
        st.error("Missing HF_TOKEN in .env")
        st.stop()

    return HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=512,
        temperature=0.7,
        huggingfacehub_api_token=hf_token
    )

# ✅ Main app logic
def main():
    st.title("Ask Wallet Chatbot (LLaMA 3)")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask your question...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            with st.spinner("Thinking..."):
                vectorstore = get_vectorstore()
                if not vectorstore:
                    return

                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt()}
                )

                response = qa_chain.invoke({'query': prompt})
                result = response.get("result", "No response from model.")
                source_documents = response.get("source_documents", [])

                source_texts = "\n\n".join([doc.page_content for doc in source_documents])
                response_display = f"{result}\n\n---\n**Source Documents:**\n{source_texts}"

                st.chat_message("assistant").markdown(response_display)
                st.session_state.messages.append({'role': 'assistant', 'content': response_display})

        except Exception as e:
            st.error("Something went wrong:")
            st.exception(e)
            print(traceback.format_exc())

if __name__ == "__main__":
    main()
