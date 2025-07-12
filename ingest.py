import os
from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader  # for PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Step 1: Load PDF
file_path = "data/resume.pdf"
loader = PyMuPDFLoader(file_path)
documents = loader.load()

# Step 2: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# Step 3: Embed + Save to FAISS
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")

print("✅ FAISS vector store created and saved locally.")
