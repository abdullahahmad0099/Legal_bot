from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os, shutil, re

# =======================
# 1. LOAD & CHUNK PDFs
# =======================
file_path = "./Dataset/"
pdf_files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.pdf')]

docs = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    pages = loader.load()
    for page in pages:
        # Add metadata for filtering
        page.metadata["source"] = os.path.basename(pdf_file)  # e.g., "2024onsc1678.pdf"
        page.metadata["page"] = page.metadata.get("page", None)
        docs.append(page)

print(f"✅ Total pages loaded: {len(docs)}")

# Better chunking for legal text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3500,
    chunk_overlap=250,
    length_function=len,
    is_separator_regex=False
)
chunks = text_splitter.split_documents(docs)
print(f"✅ Total chunks created: {len(chunks)}")

# =======================
# 2. CREATE VECTOR DB
# =======================
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

persist_directory = "./vector_db1"
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

print("✅ Vector DB created with metadata for multiple files")

# =======================
# 3. AUTO-DETECT CASE FILE FROM QUERY
# =======================
def get_case_file_from_query(query, available_files):
    """
    Extract case number like '2024 ONSC 1678' from query,
    match it to a PDF filename in the database.
    """
    match = re.search(r"(20\d{2})[- ]ONSC[- ](\d+)", query, re.IGNORECASE)
    if match:
        year, number = match.groups()
        possible_name = f"{year}onsc{number}.pdf".lower()
        for f in available_files:
            if f.lower() == possible_name:
                return f
    return None

# =======================
# 4. RUN A QUERY
# =======================
query = "What was the ruling in 2024 ONSC 1678?"

# Get all file names from DB
all_files = list(set([doc.metadata["source"] for doc in chunks]))
case_file = get_case_file_from_query(query, all_files)

if case_file:
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4, "filter": {"source": case_file}}
    )
    print(f"🔍 Searching only in: {case_file}")
else:
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    print("⚠️ No specific case detected, searching all documents.")

retrieved_docs = retriever.invoke(query)

# =======================
# 5. SHOW RESULTS
# =======================
print("\n📄 Retrieved Chunks:\n")
for i, doc in enumerate(retrieved_docs, start=1):
    print(f"Chunk {i} (Page {doc.metadata.get('page')}):\n{doc.page_content}\n{'-'*80}")
