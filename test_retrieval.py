from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings

def test_retrieval(query):
    print(f"Connecting to ./chroma_db...")
    
    # 1. Initialize embeddings with the same model used for ingestion
    embeddings = HuggingFaceEndpointEmbeddings(model="nomic-ai/nomic-embed-text-v1.5")
    
    # 2. Connect to the existing local ChromaDB
    try:
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        
        # 3. Execute similarity search
        print(f"Searching for: '{query}'...")
        results = vectorstore.similarity_search(query, k=2)
        
        # 4. Display results
        if not results:
            print("No matching results found.")
            return

        print(f"\nFound {len(results)} matches:")
        for i, doc in enumerate(results):
            print(f"\n--- Matching Chunk {i+1} ---")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Content Sneak Peek:\n{doc.page_content[:500]}...")
            print("-" * 40)
            
    except Exception as e:
        print(f"Error connecting to database: {e}")

if __name__ == "__main__":
    # Hardcoded query as requested
    test_query = "What is FastAPI?"
    test_retrieval(test_query)
