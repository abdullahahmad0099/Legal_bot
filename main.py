# main.py
from retriever_utils import retrieve_chunks

query = "What was the ruling in 2024 ONSC 1685?"
docs = retrieve_chunks(query, k=6)

print("\n📄 Retrieved Chunks:\n")
for i, doc in enumerate(docs, start=1):
    print(f"Chunk {i} (Page {doc.metadata.get('page')}):\n{doc.page_content}\n{'-'*80}")
