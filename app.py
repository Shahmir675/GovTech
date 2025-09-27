import streamlit as st
import os
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from vector_store import QdrantVectorStore
from gemini_client import GeminiClient
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="KPK Local Government Act 2013 - RAGBot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.bot {
        background-color: #475063;
    }
    .chat-message .message {
        flex: 1;
        padding-left: 1rem;
    }
    .sidebar-info {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'gemini_client' not in st.session_state:
        st.session_state.gemini_client = None
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False

def setup_connections():
    """Setup connections to Qdrant and Gemini"""
    try:
        # Get environment variables
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'kpk_local_govt_act_2013')
        
        if not all([gemini_api_key, qdrant_url, qdrant_api_key]):
            st.error("Please configure all required API keys in the .env file")
            return False
        
        # Initialize clients
        if st.session_state.vector_store is None:
            with st.spinner("Connecting to Qdrant Cloud and loading embedding model..."):
                st.session_state.vector_store = QdrantVectorStore(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    collection_name=collection_name
                )
        
        if st.session_state.gemini_client is None:
            with st.spinner("Initializing Gemini client..."):
                st.session_state.gemini_client = GeminiClient(gemini_api_key)
        
        return True
    except Exception as e:
        st.error(f"Error setting up connections: {str(e)}")
        return False

def load_documents():
    """Load and process the PDF document"""
    if st.session_state.documents_loaded:
        return True
    
    try:
        pdf_path = "THE_KHYBER_PAKHTUNKHWA_LOCAL_GOVERNMENT_ACT_2013.pdf"
        
        if not os.path.exists(pdf_path):
            st.error(f"PDF file not found: {pdf_path}")
            return False
        
        with st.spinner("Processing PDF document..."):
            # Process PDF
            processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
            chunks = processor.process_pdf(pdf_path)
            
            # Add documents to vector store
            st.session_state.vector_store.add_documents(chunks)
            st.session_state.documents_loaded = True
            
            st.success(f"Successfully processed {len(chunks)} document chunks!")
            return True
            
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return False

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📚 KPK Local Government Act 2013 - RAGBot</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ Configuration")
        
        # Setup connections
        if st.button("🔌 Connect to Services", type="primary"):
            if setup_connections():
                st.success("✅ Connected successfully!")
            else:
                st.error("❌ Connection failed!")
        
        # Load documents
        if st.session_state.vector_store is not None and not st.session_state.documents_loaded:
            if st.button("📄 Load PDF Document", type="primary"):
                load_documents()
        
        # Collection management
        if st.session_state.vector_store is not None:
            st.markdown("### 🔧 Collection Management")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Recreate Collection"):
                    with st.spinner("Recreating collection..."):
                        if st.session_state.vector_store.recreate_collection():
                            st.session_state.documents_loaded = False
                            st.success("✅ Collection recreated!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to recreate collection")
            
            with col2:
                if st.button("🗑️ Clear Collection"):
                    with st.spinner("Clearing collection..."):
                        if st.session_state.vector_store.delete_collection():
                            st.session_state.documents_loaded = False
                            st.success("✅ Collection cleared!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to clear collection")
        
        # Collection info
        if st.session_state.vector_store is not None:
            st.markdown("### 📊 Collection Status")
            info = st.session_state.vector_store.get_collection_info()
            if 'error' not in info:
                st.markdown(f"""
                <div class="sidebar-info">
                    <strong>Status:</strong> {info.get('status', 'Unknown')}<br>
                    <strong>Documents:</strong> {info.get('points_count', 0)}<br>
                    <strong>Vectors:</strong> {info.get('vectors_count', 0)}<br>
                    <strong>Embedding Model:</strong> Sentence Transformers
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"Collection error: {info['error']}")
        
        # Sample queries
        st.markdown("### 💡 Sample Queries")
        sample_queries = [
            "What is the purpose of this Act?",
            "What are the powers of local government?",
            "How are local councils constituted?",
            "What are the functions of Union Councils?",
            "What is the role of District Government?"
        ]
        
        for query in sample_queries:
            if st.button(query, key=f"sample_{hash(query)}"):
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
    
    # Main chat interface
    if st.session_state.vector_store is None or st.session_state.gemini_client is None:
        st.warning("⚠️ Please connect to services first using the sidebar.")
        return
    
    if not st.session_state.documents_loaded:
        st.warning("⚠️ Please load the PDF document first using the sidebar.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about the KPK Local Government Act 2013..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                try:
                    # Search for relevant documents
                    search_results = st.session_state.vector_store.search(prompt, limit=5)
                    
                    if not search_results:
                        response = "I couldn't find relevant information in the document. Please try rephrasing your question."
                    else:
                        # Generate response using Gemini
                        response = st.session_state.gemini_client.generate_response(prompt, search_results)
                    
                    st.markdown(response)
                    
                    # Show sources
                    if search_results:
                        with st.expander("📖 View Sources"):
                            for i, result in enumerate(search_results):
                                st.markdown(f"**Source {i+1}** (Relevance: {result['score']:.3f})")
                                st.markdown(f"```\n{result['text'][:300]}...\n```")
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    response = error_msg
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()