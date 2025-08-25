from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Folder containing PDFs
file_path = "./Dataset/"
pdf_files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.pdf')]
print(f"Found {len(pdf_files)} PDF files.")

# Load all documents from all PDFs
docs = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    pages = loader.load()
    docs.extend(pages)  # Add all pages to docs

print(f"Total pages loaded: {len(docs)}")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False
)
chunks = text_splitter.split_documents(docs)
print(f"Total chunks created: {len(chunks)}")

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create or overwrite Chroma database
persist_directory = "./vector_db"


vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Query the retriever
retrieved_docs = retriever.invoke("What was the ruling in 2024ONSC1678?")

# Print retrieved chunks
print("\nRetrieved Chunks:\n")
for i, doc in enumerate(retrieved_docs, start=1):
    print(f"Chunk {i}:\n{doc.page_content}\n{'-'*80}")
