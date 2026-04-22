from langchain_chroma import Chroma
from remote_embeddings import RemoteEmbeddings
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_tavily import TavilySearch
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.document_loaders import WebBaseLoader
import os

from dotenv import load_dotenv
from langsmith import traceable

# Load environment variables from .env file
load_dotenv()

# Initialize common components using environment variables
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
VECTORDB_DIR = os.getenv("VECTORDB_DIR", "./chroma_db")
TOP_K = int(os.getenv("TOP_K", "8"))
RE_RANK_K = int(os.getenv("RE_RANK_K", "5"))

# Use the remote embedding service URL from .env
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8000")

# Initialize the remote client
embeddings = RemoteEmbeddings(service_url=EMBEDDING_SERVICE_URL)


@tool
@traceable()
def search_web(query: str) -> str:
    """
    Search the web for real-time information or documentation.
    Returns structured results including URL, Title, and a Snippet.
    Use this to identify official documentation URLs for ingestion.
    """
    search = TavilySearch(
        max_results=15,
        search_depth="advanced",
        include_answer=False,
        include_raw_content=False,
        include_images=False,
    )
    
    try:
        raw_response = search.invoke(query)
        # Fix: The official TavilySearch returns a dict with a 'results' key
        results = raw_response.get("results", [])
        
        formatted_results = []
        source_metadata = []
        
        for i, res in enumerate(results, 1):
            url = res.get("url", "N/A")
            title = res.get("title", "Untitled")
            content = res.get("content", "No snippet available.")
            
            formatted_results.append(
                f"{i}. [WEB] {title}\n"
                f"   URL: {url}\n"
                f"   Snippet: {content[:300]}..."
            )
            # Collect clean metadata for the 'sources' state
            source_metadata.append({"title": title, "url": url})
        
        output = "\n".join(formatted_results)
        
        # Return structured data for the custom tool node
        return {
            "content": output,
            "sources": source_metadata
        }
    except Exception as e:
        return {"content": f"Error performing web search: {str(e)}", "sources": []}

@tool
@traceable()
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
        
        # Real-World Refinement: Extract Title from metadata for better RAG context
        page_title = raw_docs[0].metadata.get("title", url)
        
        # Transform HTML to clean Markdown-like text
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(raw_docs)

        # 4. Split text into manageable chunks
        # Real-World Refinement: Use Markdown-aware splitting to preserve headers and code blocks
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        splits_by_header = markdown_splitter.split_text(docs_transformed[0].page_content)

        # Further split large sections while keeping header context
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(splits_by_header)

        # 5. Ensure every chunk has the source URL and the Page Title in metadata
        for split in splits:
            split.metadata["source"] = url
            split.metadata["title"] = page_title

        # 6. Save to ChromaDB
        vectorstore.add_documents(documents=splits)
        
        # Return result
        return {
            "content": f"Successfully ingested '{url}'. Added {len(splits)} chunks to the local database.",
            "sources": [] # Ingestion doesn't add to the 'current query sources' list unless it was also a search result
        }

    except Exception as e:
        return {"content": f"Failed to ingest URL '{url}': {str(e)}", "sources": []}

@tool
@traceable()
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
    except Exception as e:
        # Fallback to standard search if re-ranking fails
        print(f"Re-ranking failed, falling back to standard search: {e}")
        docs = vectorstore.similarity_search(query, k=RE_RANK_K)
    
    if not docs:
        return {"content": "No relevant local documentation found.", "sources": []}

    # Format output with clear source indicators
    formatted_results = []
    source_metadata = []
    for i, doc in enumerate(docs, 1):
        source_url = doc.metadata.get("source", "Unknown Source")
        # Use the captured Page Title if available, otherwise fallback to filename
        title = doc.metadata.get("title", os.path.basename(source_url) if "/" in source_url else source_url)
        content = doc.page_content.strip()
        
        formatted_results.append(f"--- Local Doc {i} ---\nTitle: {title}\nSource: {source_url}\nContent: {content}")
        source_metadata.append({"title": title, "url": source_url})
    
    return {
        "content": "\n\n".join(formatted_results),
        "sources": source_metadata
    }

