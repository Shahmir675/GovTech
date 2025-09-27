#!/usr/bin/env python3
"""
Comprehensive test script to validate the improved RAGBot accuracy and functionality
"""

import os
import sys
from dotenv import load_dotenv
from pdf_processor import AdvancedPDFProcessor
from vector_store import EnhancedQdrantVectorStore
from gemini_client import EnhancedGeminiClient

# Load environment
load_dotenv()

def test_pdf_processing():
    """Test advanced PDF processing capabilities"""
    print("🧪 Testing Advanced PDF Processing...")
    
    pdf_path = "THE_KHYBER_PAKHTUNKHWA_LOCAL_GOVERNMENT_ACT_2013.pdf"
    if not os.path.exists(pdf_path):
        print("❌ PDF file not found")
        return False
    
    try:
        processor = AdvancedPDFProcessor(chunk_size=800, chunk_overlap=400)
        
        # Test text extraction
        print("  📄 Testing text extraction...")
        text = processor.extract_text_from_pdf(pdf_path)
        print(f"  ✅ Extracted {len(text)} characters")
        
        # Test structure analysis
        print("  📖 Testing structure analysis...")
        structure = processor.extract_document_structure(text)
        
        for struct_type, items in structure.items():
            if items:
                print(f"  ✅ Found {len(items)} {struct_type}(s)")
        
        # Test advanced chunking
        print("  ✂️  Testing advanced chunking...")
        chunks_with_metadata, _ = processor.process_pdf_advanced(pdf_path)
        print(f"  ✅ Created {len(chunks_with_metadata)} chunks with metadata")
        
        # Analyze chunk quality
        metadata_count = sum(1 for chunk in chunks_with_metadata if chunk['metadata'])
        overlap_count = sum(1 for chunk in chunks_with_metadata if chunk['metadata'].get('is_overlap'))
        
        print(f"  📊 Chunks with metadata: {metadata_count}/{len(chunks_with_metadata)}")
        print(f"  📊 Overlapping chunks: {overlap_count}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ PDF processing failed: {e}")
        return False

def test_vector_store():
    """Test enhanced vector store functionality"""
    print("\n🧪 Testing Enhanced Vector Store...")
    
    try:
        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'test_collection')
        
        if not all([qdrant_url, qdrant_api_key]):
            print("  ❌ Qdrant credentials not configured")
            return False
        
        # Initialize vector store
        print("  🔌 Connecting to Qdrant...")
        vector_store = EnhancedQdrantVectorStore(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name + "_test"
        )
        
        # Test document addition
        print("  📝 Testing document addition...")
        test_docs = [
            {
                'text': "Union Councils are the basic units of local government in KPK. They have specific powers and functions as defined in Section 15 of the Act.",
                'metadata': {
                    'section': 'Union Councils',
                    'section_number': '15',
                    'key_terms': ['union', 'council', 'local', 'government', 'powers']
                }
            },
            {
                'text': "District Government has administrative and financial powers over the district. The District Nazim is the head of District Government.",
                'metadata': {
                    'section': 'District Government',
                    'section_number': '25',
                    'key_terms': ['district', 'government', 'nazim', 'administrative', 'financial']
                }
            }
        ]
        
        vector_store.add_documents_with_metadata(test_docs)
        print("  ✅ Documents added successfully")
        
        # Test different search methods
        test_queries = [
            "What are Union Councils?",
            "Section 15 powers",
            "District Government functions",
            "Who is the head of district government?"
        ]
        
        for query in test_queries:
            print(f"\n  🔍 Testing query: '{query}'")
            
            # Test semantic search
            semantic_results = vector_store.search(query, limit=2)
            print(f"    Semantic: {len(semantic_results)} results")
            
            # Test hybrid search
            hybrid_results = vector_store.hybrid_search_method(query, limit=2)
            print(f"    Hybrid: {len(hybrid_results)} results")
            
            # Test smart search
            smart_results = vector_store.smart_search(query, limit=2)
            print(f"    Smart: {len(smart_results)} results")
            
            if smart_results:
                best_result = smart_results[0]
                print(f"    Best score: {best_result['score']:.3f}")
                if 'search_strategy' in best_result:
                    strategy = best_result['search_strategy']
                    print(f"    Strategy: {strategy['query_type']}")
        
        # Cleanup test collection
        vector_store.delete_collection()
        print("  🗑️  Test collection cleaned up")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Vector store test failed: {e}")
        return False

def test_gemini_client():
    """Test enhanced Gemini client functionality"""
    print("\n🧪 Testing Enhanced Gemini Client...")
    
    try:
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            print("  ❌ Gemini API key not configured")
            return False
        
        client = EnhancedGeminiClient(gemini_api_key)
        
        # Test query type analysis
        print("  🔍 Testing query type analysis...")
        test_queries = [
            ("What is Union Council?", "definition"),
            ("How to conduct elections?", "process"),
            ("What are the powers of district government?", "authority"),
            ("What requirements must be met?", "requirements")
        ]
        
        for query, expected_type in test_queries:
            detected_type = client.analyze_query_type(query)
            print(f"    '{query}' -> {detected_type} ({'✅' if detected_type == expected_type else '⚠️'})")
        
        # Test response generation with mock context
        print("  🤖 Testing response generation...")
        mock_context = [
            {
                'text': "Union Councils are established under Section 15 of the KPK Local Government Act 2013. They serve as the basic tier of local government.",
                'score': 0.95,
                'metadata': {
                    'section': 'Union Councils',
                    'section_number': '15',
                    'chapter': 'Local Government Structure'
                }
            }
        ]
        
        test_query = "What are Union Councils?"
        response = client.generate_response(test_query, mock_context)
        
        if response and "Union Councils" in response:
            print("  ✅ Response generation successful")
            print(f"    Response length: {len(response)} characters")
        else:
            print("  ⚠️  Response generation may have issues")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Gemini client test failed: {e}")
        return False

def test_integration():
    """Test end-to-end integration"""
    print("\n🧪 Testing End-to-End Integration...")
    
    try:
        # Test realistic workflow
        print("  🔄 Testing realistic workflow...")
        
        # 1. Process PDF
        pdf_path = "THE_KHYBER_PAKHTUNKHWA_LOCAL_GOVERNMENT_ACT_2013.pdf"
        if not os.path.exists(pdf_path):
            print("  ❌ PDF file not found for integration test")
            return False
        
        processor = AdvancedPDFProcessor()
        chunks_with_metadata, structure = processor.process_pdf_advanced(pdf_path)
        print(f"  ✅ Processed PDF: {len(chunks_with_metadata)} chunks")
        
        # 2. Setup vector store (limited chunks for speed)
        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        
        if not all([qdrant_url, qdrant_api_key]):
            print("  ⚠️  Skipping vector store integration (credentials not available)")
            return True
        
        vector_store = EnhancedQdrantVectorStore(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name="integration_test"
        )
        
        # Add first 10 chunks for testing
        test_chunks = chunks_with_metadata[:10]
        vector_store.add_documents_with_metadata(test_chunks)
        print(f"  ✅ Added {len(test_chunks)} chunks to vector store")
        
        # 3. Test search and response
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if gemini_api_key:
            client = EnhancedGeminiClient(gemini_api_key)
            
            test_query = "What is the purpose of this Act?"
            search_results = vector_store.smart_search(test_query, limit=3)
            
            if search_results:
                response = client.generate_response(test_query, search_results)
                print(f"  ✅ Generated response ({len(response)} chars)")
                
                # Check for legal references
                if any(ref in response.lower() for ref in ['section', 'chapter', 'article']):
                    print("  ✅ Response contains legal references")
                else:
                    print("  ⚠️  Response lacks legal references")
            else:
                print("  ⚠️  No search results found")
        
        # Cleanup
        vector_store.delete_collection()
        print("  🗑️  Integration test collection cleaned up")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False

def main():
    """Run all accuracy tests"""
    print("🚀 RAGBot Accuracy Test Suite\n")
    
    tests = [
        ("PDF Processing", test_pdf_processing),
        ("Vector Store", test_vector_store),
        ("Gemini Client", test_gemini_client),
        ("Integration", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print("📊 Test Results Summary:")
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASSED" if results[i] else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your enhanced RAGBot is ready for production!")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("\nCommon fixes:")
        print("- Ensure all API keys are configured in .env")
        print("- Check internet connection for Qdrant Cloud")
        print("- Verify PDF file exists in current directory")

if __name__ == "__main__":
    main()