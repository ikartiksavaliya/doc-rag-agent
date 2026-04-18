# Required: pip install langchain-community langchain-ollama langchain-chroma chromadb html2text

import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

def ingest_docs(url, persist_directory="./chroma_db"):
    print(f"Scraping {url}...")
    
    # 1. Load raw HTML
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    if not docs:
        print("No documents loaded.")
        return

    print(f"Loaded {len(docs)} document(s). Converting to Markdown...")

    # 2. Transform HTML to clean Markdown
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)

    # 3. Split Markdown into manageable chunks
    print("Splitting markdown into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        add_start_index=True
    )
    splits = text_splitter.split_documents(docs_transformed)
    print(f"Created {len(splits)} chunks.")

    # 4. Use OllamaEmbeddings with nomic-embed-text
    print("Initializing embeddings with Ollama (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 5. Store chunks in local ChromaDB
    print(f"Storing chunks in {persist_directory}...")
    # Clean up existing DB for a fresh start
    if os.path.exists(persist_directory):
        import shutil
        print(f"Cleaning up existing database at {persist_directory}...")
        shutil.rmtree(persist_directory)
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Ingested {len(splits)} chunks into {persist_directory}")
    print("Ingested content sample:", splits[0].page_content[:150], "...")
    return vectorstore

if __name__ == "__main__":
    # Target URL can be any framework documentation
    target_url = "https://fastapi.tiangolo.com/"
    ingest_docs(target_url)
