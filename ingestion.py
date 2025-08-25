# ingestion.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os, shutil

# Paths
file_path = "./Dataset/"
persist_directory = "./vector_db1"

# Load PDFs
pdf_files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.pdf')]
docs = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    pages = loader.load()
    for page in pages:
        page.metadata["source"] = os.path.basename(pdf_file)
        page.metadata["page"] = page.metadata.get("page", None)
        docs.append(page)

print(f"✅ Total pages loaded: {len(docs)}")

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3500,
    chunk_overlap=250
)
chunks = text_splitter.split_documents(docs)
print(f"✅ Total chunks created: {len(chunks)}")

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Clear old DB
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

# Store in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

print(f"✅ Vector DB created at {persist_directory}")
