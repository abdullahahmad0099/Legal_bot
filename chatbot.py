import os
from dotenv import load_dotenv
from retriever_utils import retrieve_chunks
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # You can change model
    temperature=0.3,
    api_key=groq_api_key
)

def run_chat():
    print("🤖 LLM Chatbot Ready! Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        # Retrieve context from vector DB
        context_docs = retrieve_chunks(query, k=2)

        # Merge retrieved text into a single context string
        context_text = "\n\n".join([doc.page_content for doc in context_docs])

        # Build final prompt for LLM
        prompt = f"""You are a helpful legal assistant.
Answer the user's question based only on the following context:

{context_text}

Question: {query}
Answer:"""

        # Get LLM response
        response = llm.invoke(prompt)
        print(f"\nAI: {response.content}")

        # Show sources
        print("\n📄 Sources:")
        for doc in context_docs:
            print(f"- {doc.metadata.get('source')} (Page {doc.metadata.get('page')})")

if __name__ == "__main__":
    run_chat()
