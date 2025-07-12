import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load vector DB + model
vectorstore = FAISS.load_local("faiss_index", OpenAIEmbeddings(openai_api_key=openai_api_key))
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=openai_api_key)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Streamlit App UI
st.set_page_config(page_title="AI Resume Chatbot", page_icon="🤖")
st.title("🧠 AI Chatbot - Ask Anything About Your Resume")

query = st.text_input("Ask a question about your resume:")

if query:
    with st.spinner("Thinking..."):
        answer = qa.run(query)
        st.success(answer)
