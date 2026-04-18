from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

@tool
def search_documentation(query: str) -> str:
    """
    Search for relevant information in the local documentation vector store.
    Use this tool to find technical details, API references, or tutorials 
    stored in the database.
    
    Args:
        query: The search query to look up in the documentation.
        
    Returns:
        A combined string containing the top 3 matching chunks found in the database.
    """
    # 1. Initialize the embedding model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # 2. Connect to the existing local ChromaDB
    # We use langchain_chroma for modern compatibility
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    # 3. Perform similarity search for the top 3 results
    docs = vectorstore.similarity_search(query, k=3)
    
    # 4. Join the results into a single string for the agent to process
    return "\n\n---\n\n".join([doc.page_content for doc in docs])
