from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Initialize common components
embeddings = OllamaEmbeddings(model="nomic-embed-text")
VECTORDB_DIR = "./chroma_db"

@tool
def search_web(query: str) -> str:
    """
    Search the web for real-time information or documentation.
    CRITICAL: Do not pass the user's raw conversational question into this tool. 
    You must extract the core technical concepts and generate a highly optimized 
    search engine query (e.g., instead of "how do I do x in y", use "framework Y feature X documentation").
    """
    search = DuckDuckGoSearchResults()
    results = search.run(query)
    
    return f"Web Search Results for '{query}':\n\n{results}"

@tool
def ingest_url(url: str) -> str:
    """
    Autonomously ingest a documentation URL into the local vector database.
    It scrapes the page, converts it to markdown, splits it into chunks, 
    and saves it if it hasn't been ingested yet.
    """
    # 1. Initialize Vectorstore
    vectorstore = Chroma(
        persist_directory=VECTORDB_DIR,
        embedding_function=embeddings
    )

    # 2. Check for duplicates using metadata filtering
    # We check if any documents exist with this source URL
    existing_docs = vectorstore.get(where={"source": url})
    if existing_docs and existing_docs.get("ids"):
        return f"Deduplication System: The URL '{url}' has already been ingested. Skipping to prevent duplicates."

    try:
        # 3. Load and Transform (Scrape -> Markdown)
        loader = WebBaseLoader(url)
        raw_docs = loader.load()
        
        # Transform HTML to clean Markdown-like text
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(raw_docs)

        # 4. Split text into manageable chunks
        # Using 1000/200 as requested for 'best' defaults
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs_transformed)

        # 5. Ensure every chunk has the source URL in metadata
        for split in splits:
            split.metadata["source"] = url

        # 6. Save to ChromaDB
        vectorstore.add_documents(documents=splits)
        
        return f"Successfully ingested '{url}'. Added {len(splits)} chunks to the local database."

    except Exception as e:
        return f"Failed to ingest URL '{url}': {str(e)}"

@tool
def search_local_docs(query: str) -> str:
    """
    Search the local documentation vector store for technical answers.
    Returns chunks of text along with their source URLs for easy citation.
    """
    vectorstore = Chroma(
        persist_directory=VECTORDB_DIR,
        embedding_function=embeddings
    )
    
    # Perform similarity search
    docs = vectorstore.similarity_search(query, k=3)
    
    if not docs:
        return "No relevant local documentation found."

    # Format output with clear source indicators
    formatted_results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown Source")
        content = doc.page_content.strip()
        formatted_results.append(f"--- Local Doc {i} ---\nSource: {source}\nContent: {content}")
    
    return "\n\n".join(formatted_results)
