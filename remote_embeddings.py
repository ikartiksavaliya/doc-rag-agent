import requests
from typing import List
from langchain_core.embeddings import Embeddings

class RemoteEmbeddings(Embeddings):
    """
    Client for the standalone FastAPI embedding service.
    Implements the standard LangChain Embeddings interface.
    """
    
    def __init__(self, service_url: str):
        self.service_url = service_url.rstrip("/")
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        try:
            response = requests.post(
                f"{self.service_url}/embed_documents",
                json={"texts": texts},
                timeout=60
            )
            response.raise_for_status()
            return response.json()["embeddings"]
        except Exception as e:
            print(f"Error calling remote embedding service: {e}")
            raise
            
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        try:
            response = requests.post(
                f"{self.service_url}/embed_query",
                json={"text": text},
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            print(f"Error calling remote embedding service: {e}")
            raise
