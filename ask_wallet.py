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

def get_vectorstore():
    try:
        st.info("Loading vectorstore...")
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error("Failed to load FAISS vector store.")
        st.exception(e)
        print(traceback.format_exc())
        return None

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    if not HF_TOKEN:
        st.error("Missing `HF_TOKEN` in your .env file or environment variables.")
        st.stop()

    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        max_new_tokens=150,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN
    )

def main():
    st.title("Ask Wallet Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know — don't try to make up an answer. 
        Do not provide anything outside the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        # ✅ Valid open-source model for text-generation
        # HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3" # old
        HUGGINGFACE_REPO_ID = "tiiuae/falcon-7b-instruct"
        # HUGGINGFACE_REPO_ID = "google/flan-t5-xxl"

        try:
            with st.spinner("Generating response..."):
                # ✅ Debug prints (also visible in terminal)
                print("Prompt entered:", prompt)
                print("Using model:", HUGGINGFACE_REPO_ID)
                print("HF_TOKEN present:", HF_TOKEN is not None)

                vectorstore = get_vectorstore()
                if not vectorstore:
                    return

                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke({'query': prompt})
                result = response.get("result", "No response from model.")
                source_documents = response.get("source_documents", [])

                # ✅ Clean formatting of source docs
                source_texts = "\n\n".join([doc.page_content for doc in source_documents])
                result_to_show = f"{result}\n\n---\n**Source Documents:**\n{source_texts}"

                st.chat_message('assistant').markdown(result_to_show)
                st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error("An unexpected error occurred:")
            st.exception(e)
            print(traceback.format_exc())

if __name__ == "__main__":
    main()
