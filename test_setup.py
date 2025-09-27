#!/usr/bin/env python3
"""
Simple test script to verify the RAGBot setup works correctly
"""

import os
import sys
from dotenv import load_dotenv

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import google.generativeai as genai
        print("✅ Google Generative AI imported successfully (for chat only)")
    except ImportError as e:
        print(f"❌ Google Generative AI import failed: {e}")
        return False
    
    try:
        from qdrant_client import QdrantClient
        print("✅ Qdrant client imported successfully")
    except ImportError as e:
        print(f"❌ Qdrant client import failed: {e}")
        return False
    
    try:
        import PyPDF2
        print("✅ PyPDF2 imported successfully")
    except ImportError as e:
        print(f"❌ PyPDF2 import failed: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Sentence Transformers import failed: {e}")
        return False
    
    return True

def test_environment():
    """Test if environment variables are configured"""
    print("\n🔧 Testing environment configuration...")
    
    load_dotenv()
    
    configured_count = 0
    
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key and gemini_key != 'your_gemini_api_key_here':
        print("✅ Gemini API key configured")
        configured_count += 1
    else:
        print("⚠️  Gemini API key not configured")
    
    qdrant_url = os.getenv('QDRANT_URL')
    if qdrant_url and qdrant_url != 'your_qdrant_cloud_url_here':
        print("✅ Qdrant URL configured")
        configured_count += 1
    else:
        print("⚠️  Qdrant URL not configured")
    
    qdrant_key = os.getenv('QDRANT_API_KEY')
    if qdrant_key and qdrant_key != 'your_qdrant_api_key_here':
        print("✅ Qdrant API key configured")
        configured_count += 1
    else:
        print("⚠️  Qdrant API key not configured")
    
    return configured_count == 3

def test_modules():
    """Test if our custom modules can be imported"""
    print("\n📦 Testing custom modules...")
    
    try:
        from pdf_processor import PDFProcessor
        print("✅ PDFProcessor imported successfully")
    except ImportError as e:
        print(f"❌ PDFProcessor import failed: {e}")
        return False
    
    try:
        from vector_store import QdrantVectorStore
        print("✅ QdrantVectorStore imported successfully")
    except ImportError as e:
        print(f"❌ QdrantVectorStore import failed: {e}")
        return False
    
    try:
        from gemini_client import GeminiClient
        print("✅ GeminiClient imported successfully")
    except ImportError as e:
        print(f"❌ GeminiClient import failed: {e}")
        return False
    
    return True

def test_pdf_file():
    """Test if the PDF file exists"""
    print("\n📄 Testing PDF file...")
    
    pdf_path = "THE_KHYBER_PAKHTUNKHWA_LOCAL_GOVERNMENT_ACT_2013.pdf"
    if os.path.exists(pdf_path):
        file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # Size in MB
        print(f"✅ PDF file found ({file_size:.1f} MB)")
        return True
    else:
        print(f"❌ PDF file not found: {pdf_path}")
        return False

def main():
    """Run all tests"""
    print("🚀 RAGBot Setup Test\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Environment Test", test_environment),
        ("Module Test", test_modules),
        ("PDF File Test", test_pdf_file),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result if result is not None else False)
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "="*50)
    print("📊 Test Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All tests passed! Your RAGBot setup is ready!")
        print("\nTo run the application:")
        print("   streamlit run app.py")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        print("\nTo install missing dependencies:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()