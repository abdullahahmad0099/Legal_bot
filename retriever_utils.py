# retriever_utils.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import re

PERSIST_DIR = "./vector_db1"

# Load embeddings (must match ingestion)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load vectorstore
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

# Helper: detect case from query
def get_case_file_from_query(query, available_files):
    match = re.search(r"(20\d{2})[- ]ONSC[- ](\d+)", query, re.IGNORECASE)
    if match:
        year, number = match.groups()
        possible_name = f"{year}onsc{number}.pdf".lower()
        for f in available_files:
            if f.lower() == possible_name:
                return f
    return None

# Main retrieval function
def retrieve_chunks(query, k=4):
    all_files = list(set([md["source"] for md in vectorstore.get(include=["metadatas"])["metadatas"]]))
    case_file = get_case_file_from_query(query, all_files)

    if case_file:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k, "filter": {"source": case_file}})
        print(f"🔍 Searching only in: {case_file}")
    else:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        print("⚠️ No specific case detected, searching all documents.")

    return retriever.invoke(query)
