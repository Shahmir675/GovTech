import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class VectorStore:
    def __init__(self):
        self.qdrant_url = os.getenv('QDRANT_URL')
        self.qdrant_api_key = os.getenv('QDRANT_API_KEY')
        self.collection_name = os.getenv('COLLECTION_NAME', 'kp_local_government_act')
        self.vector_size = int(os.getenv('VECTOR_SIZE', 768))
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )
        
        # Initialize embedding model
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def create_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                print(f"Created collection: {self.collection_name}")
            else:
                print(f"Collection {self.collection_name} already exists")
        except Exception as e:
            print(f"Error creating collection: {e}")
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Convert texts to embeddings"""
        embeddings = self.encoder.encode(texts)
        return embeddings
    
    def add_documents(self, texts: List[str], metadata: List[Dict[str, Any]] = None):
        """Add documents to vector store"""
        if not texts:
            return
        
        embeddings = self.encode_texts(texts)
        
        points = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            payload = {
                "text": text,
                "chunk_id": i
            }
            
            # Add metadata if provided
            if metadata and i < len(metadata):
                payload.update(metadata[i])
            
            point = models.PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)
        
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Added {len(points)} documents to collection")
        except Exception as e:
            print(f"Error adding documents: {e}")
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            query_embedding = self.encode_texts([query])[0]
            
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit
            )
            
            results = []
            for point in search_result:
                results.append({
                    "text": point.payload["text"],
                    "score": point.score,
                    "chunk_id": point.payload.get("chunk_id", 0)
                })
            
            return results
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def delete_collection(self):
        """Delete the collection"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def get_collection_info(self):
        """Get information about the collection"""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return info
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return None