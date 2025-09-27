from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any
import numpy as np
import uuid
from sentence_transformers import SentenceTransformer

class QdrantVectorStore:
    def __init__(self, url: str, api_key: str, collection_name: str):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        
        # Initialize Sentence Transformer embedding model
        try:
            print("Loading Sentence Transformer model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
            print("✅ Sentence Transformer model loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load Sentence Transformer model: {e}")
        
        # Create collection if it doesn't exist
        self._create_collection_if_not_exists()
    
    def _create_collection_if_not_exists(self):
        """Create collection with appropriate vector configuration"""
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,  # Dynamic embedding dimension
                    distance=Distance.COSINE
                )
            )
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Sentence Transformers"""
        try:
            print(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self.embedding_model.encode(texts).tolist()
            print(f"✅ Successfully generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Return zero vectors as fallback
            return [[0.0] * self.embedding_dim] * len(texts)
    
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """Add documents to the vector store"""
        if metadatas is None:
            metadatas = [{"text": text, "chunk_id": i} for i, text in enumerate(texts)]
        
        # Generate embeddings
        embeddings = self.get_embeddings(texts)
        
        # Create points for Qdrant
        points = []
        for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": text,
                    "metadata": metadata
                }
            )
            points.append(point)
        
        # Upsert points to collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        # Generate query embedding
        query_embedding = self.get_embeddings([query])[0]
        
        # Search in Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
        # Format results
        results = []
        for result in search_results:
            results.append({
                "text": result.payload["text"],
                "score": result.score,
                "metadata": result.payload.get("metadata", {})
            })
        
        return results
    
    def get_collection_info(self):
        """Get information about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "status": info.status,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count
            }
        except Exception as e:
            return {"error": str(e)}