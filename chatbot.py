import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load vector store
vectorstore = FAISS.load_local("faiss_index", OpenAIEmbeddings(openai_api_key=openai_api_key))

# Set up LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=openai_api_key)

# Create retrieval-based QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# CLI loop
print("🤖 Ask anything about your document (type 'exit' to quit)")
while True:
    query = input("🧠 You: ")
    if query.lower() == 'exit':
        print("👋 Exiting...")
        break
    response = qa.run(query)
    print("🤖 Bot:", response)
