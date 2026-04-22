import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
DEVICE = "cuda" if os.getenv("USE_CUDA", "true").lower() == "true" else "cpu"

# Global model instance
embeddings = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embeddings
    print(f"--- Loading Embedding Model: {EMBEDDING_MODEL} on {DEVICE} ---")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': DEVICE},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 64}
    )
    print("--- Model Loaded Successfully ---")
    yield
    print("--- Shutting Down Embedding Service ---")

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    text: str

class DocRequest(BaseModel):
    texts: list[str]

@app.get("/health")
async def health():
    return {"status": "ready", "model": EMBEDDING_MODEL, "device": DEVICE}

@app.post("/embed_query")
async def embed_query(request: QueryRequest):
    if not embeddings:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        vector = embeddings.embed_query(request.text)
        return {"embedding": vector}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed_documents")
async def embed_documents(request: DocRequest):
    if not embeddings:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        vectors = embeddings.embed_documents(request.texts)
        return {"embeddings": vectors}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
