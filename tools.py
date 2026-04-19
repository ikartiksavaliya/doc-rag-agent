from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
import os

from dotenv import load_dotenv
from logger import agent_logger

# Load environment variables from .env file
load_dotenv()

# Initialize common components using environment variables
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
VECTORDB_DIR = os.getenv("VECTORDB_DIR", "./chroma_db")
TOP_K = int(os.getenv("TOP_K", "5"))
RE_RANK_K = int(os.getenv("RE_RANK_K", "3"))

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

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
    
    agent_logger.log("tool_call", {"tool": "search_web", "query": query, "result_summary": results[:200]})
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
        
        agent_logger.log("tool_call", {"tool": "ingest_url", "url": url, "chunks_added": len(splits)})
        return f"Successfully ingested '{url}'. Added {len(splits)} chunks to the local database."

    except Exception as e:
        agent_logger.log("tool_error", {"tool": "ingest_url", "url": url, "error": str(e)})
        return f"Failed to ingest URL '{url}': {str(e)}"

@tool
def search_local_docs(query: str) -> str:
    """
    Search the local documentation vector store for technical answers.
    Uses a two-stage process: similarity search followed by FlashRank re-scoring.
    """
    vectorstore = Chroma(
        persist_directory=VECTORDB_DIR,
        embedding_function=embeddings
    )
    
    # Initialize the re-ranker
    try:
        compressor = FlashrankRerank(top_n=RE_RANK_K)
        retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        
        # Perform compressed (re-ranked) search
        docs = compression_retriever.invoke(query)
        agent_logger.log("tool_call", {"tool": "search_local_docs", "query": query, "re_ranked": True, "results_count": len(docs)})
    except Exception as e:
        # Fallback to standard search if re-ranking fails
        print(f"Re-ranking failed, falling back to standard search: {e}")
        docs = vectorstore.similarity_search(query, k=RE_RANK_K)
        agent_logger.log("tool_call", {"tool": "search_local_docs", "query": query, "re_ranked": False, "results_count": len(docs), "error": str(e)})
    
    if not docs:
        return "No relevant local documentation found."

    # Format output with clear source indicators
    formatted_results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown Source")
        content = doc.page_content.strip()
        formatted_results.append(f"--- Local Doc {i} ---\nSource: {source}\nContent: {content}")
    
    return "\n\n".join(formatted_results)

