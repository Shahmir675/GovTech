#!/usr/bin/env python3
"""
Test script to verify vector dimensions are working correctly
"""

import os
from dotenv import load_dotenv

def test_vector_dimensions():
    """Test if vector dimensions match between embedding model and Qdrant"""
    print("🧪 Testing Vector Dimensions\n")
    
    # Load environment
    load_dotenv()
    
    try:
        # Test Sentence Transformers
        print("📐 Testing Sentence Transformers...")
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_text = "This is a test sentence for embedding."
        embedding = model.encode([test_text])
        
        print(f"✅ Model loaded: all-MiniLM-L6-v2")
        print(f"✅ Embedding shape: {embedding.shape}")
        print(f"✅ Vector dimension: {embedding.shape[1]}")
        
        expected_dim = 384
        actual_dim = embedding.shape[1]
        
        if actual_dim == expected_dim:
            print(f"✅ Dimensions match: {actual_dim}")
        else:
            print(f"❌ Dimension mismatch: expected {expected_dim}, got {actual_dim}")
            return False
        
        # Test Qdrant connection and collection
        print("\n🔗 Testing Qdrant connection...")
        
        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'kpk_local_govt_act_2013_st')
        
        if not all([qdrant_url, qdrant_api_key]):
            print("❌ Qdrant credentials not configured")
            return False
        
        from vector_store import QdrantVectorStore
        
        # Initialize vector store (this will create/check collection)
        vector_store = QdrantVectorStore(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name
        )
        
        print(f"✅ Connected to Qdrant")
        print(f"✅ Collection: {collection_name}")
        
        # Get collection info
        info = vector_store.get_collection_info()
        if 'error' not in info:
            print(f"✅ Collection status: {info.get('status', 'Unknown')}")
            print(f"✅ Points count: {info.get('points_count', 0)}")
        else:
            print(f"⚠️  Collection info error: {info['error']}")
        
        # Test adding a single document
        print("\n📝 Testing document addition...")
        test_texts = ["This is a test document for the KPK Local Government Act."]
        
        try:
            vector_store.add_documents(test_texts)
            print("✅ Successfully added test document")
            
            # Test search
            print("\n🔍 Testing search...")
            results = vector_store.search("test document", limit=1)
            if results:
                print(f"✅ Search successful, found {len(results)} results")
                print(f"✅ Top result score: {results[0]['score']:.4f}")
            else:
                print("⚠️  No search results found")
                
        except Exception as e:
            print(f"❌ Error with document operations: {e}")
            return False
        
        print("\n🎉 All dimension tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_vector_dimensions()
    if success:
        print("\n✅ Vector dimensions are working correctly!")
    else:
        print("\n❌ Vector dimension issues detected. Please check the errors above.")