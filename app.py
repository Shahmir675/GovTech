import streamlit as st
import os
from dotenv import load_dotenv
from pdf_processor import AdvancedPDFProcessor
from vector_store import EnhancedQdrantVectorStore
from gemini_client import EnhancedGeminiClient
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
    if 'document_structure' not in st.session_state:
        st.session_state.document_structure = None
    if 'search_weights' not in st.session_state:
        st.session_state.search_weights = None
    if 'search_mode' not in st.session_state:
        st.session_state.search_mode = "Smart Search (Recommended)"
    if 'processing_stats' not in st.session_state:
        st.session_state.processing_stats = None

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
                st.session_state.vector_store = EnhancedQdrantVectorStore(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    collection_name=collection_name
                )
        
        if st.session_state.gemini_client is None:
            with st.spinner("Initializing Gemini client..."):
                st.session_state.gemini_client = EnhancedGeminiClient(gemini_api_key)
        
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
        
        with st.spinner("Processing PDF document with advanced analysis..."):
            # Process PDF with advanced features
            processor = AdvancedPDFProcessor()
            
            # Use advanced processing
            chunks_with_metadata, structure = processor.process_pdf_advanced(pdf_path)
            
            # Add documents with metadata to vector store
            st.session_state.vector_store.add_documents_with_metadata(chunks_with_metadata)
            st.session_state.documents_loaded = True

            # Store structure for reference and processing statistics
            st.session_state.document_structure = structure
            total_chunks = len(chunks_with_metadata)
            clause_chunks = sum(1 for doc in chunks_with_metadata if doc['metadata'].get('is_clause_variant'))
            overlap_chunks = sum(1 for doc in chunks_with_metadata if doc['metadata'].get('is_overlap'))
            st.session_state.processing_stats = {
                'total_chunks': total_chunks,
                'clause_chunks': clause_chunks,
                'overlap_chunks': overlap_chunks
            }

            st.success(
                "Successfully processed {total} document representations (" \
                "{clauses} clause-level, {overlaps} contextual overlaps)!".format(
                    total=total_chunks,
                    clauses=clause_chunks,
                    overlaps=overlap_chunks
                )
            )
            
            # Show structure summary
            structure_summary = []
            for struct_type, items in structure.items():
                if items:
                    structure_summary.append(f"📋 {len(items)} {struct_type.title()}(s)")
            
            if structure_summary:
                st.info("Document Structure: " + " | ".join(structure_summary))
            
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
                            st.session_state.document_structure = None
                            st.session_state.processing_stats = None
                            st.success("✅ Collection recreated!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to recreate collection")
            
            with col2:
                if st.button("🗑️ Clear Collection"):
                    with st.spinner("Clearing collection..."):
                        if st.session_state.vector_store.delete_collection():
                            st.session_state.documents_loaded = False
                            st.session_state.document_structure = None
                            st.session_state.processing_stats = None
                            st.success("✅ Collection cleared!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to clear collection")
        
        # Collection info
        if st.session_state.vector_store is not None:
            st.markdown("### 📊 Collection Status")
            info = st.session_state.vector_store.get_collection_info()
            if 'error' not in info:
                embedding_overview = st.session_state.vector_store.get_embedding_overview()
                model_lines = [
                    f"{model['name']} (dim={model['dimension']}, weight={model['weight']})"
                    for model in embedding_overview.get('models', [])
                ]
                model_text = "<br>".join(model_lines) if model_lines else "Unknown"
                st.markdown(f"""
                <div class="sidebar-info">
                    <strong>Status:</strong> {info.get('status', 'Unknown')}<br>
                    <strong>Documents:</strong> {info.get('points_count', 0)}<br>
                    <strong>Vectors:</strong> {info.get('vectors_count', 0)}<br>
                    <strong>Embedding Dimension:</strong> {embedding_overview.get('dimension')}<br>
                    <strong>Embedding Ensemble:</strong><br>{model_text}<br>
                    <strong>Variant Weights:</strong> {embedding_overview.get('variant_weights')}<br>
                    <strong>Search Engine:</strong> Hybrid (Semantic + Keyword + Entity)
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"Collection error: {info['error']}")
        
        # Document structure info
        if hasattr(st.session_state, 'document_structure') and st.session_state.document_structure:
            st.markdown("### 📖 Document Structure")
            structure = st.session_state.document_structure
            
            structure_info = []
            for struct_type, items in structure.items():
                if items:
                    structure_info.append(f"**{struct_type.title()}:** {len(items)}")
            
            if structure_info:
                st.markdown("<br>".join(structure_info), unsafe_allow_html=True)
            if st.session_state.processing_stats:
                stats = st.session_state.processing_stats
                st.caption(
                    f"Chunks indexed: {stats['total_chunks']} | Clause variants: {stats['clause_chunks']} | Overlaps: {stats['overlap_chunks']}"
                )
        
        # Search settings
        st.markdown("### ⚙️ Search Settings")
        search_mode = st.selectbox(
            "Search Mode",
            options=["Smart Search (Recommended)", "Hybrid Search", "Semantic Search"],
            help="Smart Search automatically chooses the best strategy based on your query"
        )
        st.session_state.search_mode = search_mode

        if search_mode == "Hybrid Search":
            st.markdown("**Search Weights:**")
            col1, col2 = st.columns(2)
            with col1:
                semantic_weight = st.slider("Semantic", 0.0, 1.0, 0.4, 0.1)
                keyword_weight = st.slider("Keyword", 0.0, 1.0, 0.3, 0.1)
            with col2:
                tfidf_weight = st.slider("TF-IDF", 0.0, 1.0, 0.2, 0.1)
                entity_weight = st.slider("Entity", 0.0, 1.0, 0.1, 0.1)
            
            # Normalize weights
            total_weight = semantic_weight + keyword_weight + tfidf_weight + entity_weight
            if total_weight > 0:
                weights = {
                    'semantic': semantic_weight / total_weight,
                    'keyword': keyword_weight / total_weight,
                    'tfidf': tfidf_weight / total_weight,
                    'entity': entity_weight / total_weight
                }
                st.session_state.search_weights = weights
        else:
            st.session_state.search_weights = None
        
        # Sample queries
        st.markdown("### 💡 Sample Queries")
        sample_queries = [
            "What is the purpose of this Act?",
            "What are the powers of local government?",
            "How are local councils constituted?",
            "What are the functions of Union Councils?",
            "What is the role of District Government?",
            "Section 15 powers and functions",
            "How to conduct elections for local councils?",
            "What are the financial powers of district government?",
            "Define Union Council in this Act",
            "Procedure for budget approval"
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
        
        # Generate response with enhanced search
        with st.chat_message("assistant"):
            with st.spinner("Analyzing query and searching documents..."):
                try:
                    # Determine search method based on UI selection
                    search_mode = st.session_state.search_mode

                    if search_mode == "Smart Search (Recommended)":
                        search_results = st.session_state.vector_store.smart_search(prompt, limit=5)
                    elif search_mode == "Hybrid Search":
                        weights = getattr(st.session_state, 'search_weights', None)
                        search_results = st.session_state.vector_store.hybrid_search_method(prompt, limit=5, search_weights=weights)
                    else:  # Semantic Search
                        search_results = st.session_state.vector_store.search(prompt, limit=5)
                    
                    if not search_results:
                        response = "I couldn't find relevant information in the KPK Local Government Act 2013. Please try rephrasing your question or ask about specific topics covered in the act."
                    else:
                        # Generate enhanced response using Gemini
                        response = st.session_state.gemini_client.generate_response(prompt, search_results)
                    
                    st.markdown(response)
                    
                    # Enhanced source display
                    if search_results:
                        with st.expander("📖 View Detailed Sources", expanded=False):
                            for i, result in enumerate(search_results):
                                metadata = result.get('metadata', {})
                                score = result.get('score', 0)
                                
                                # Create source header with metadata
                                source_header = f"**Source {i+1}** (Relevance: {score:.3f})"
                                
                                # Add structural information
                                structure_info = []
                                if metadata.get('chapter'):
                                    structure_info.append(f"Chapter: {metadata['chapter']}")
                                if metadata.get('section'):
                                    structure_info.append(f"Section: {metadata['section']}")
                                if metadata.get('article'):
                                    structure_info.append(f"Article: {metadata['article']}")
                                
                                if structure_info:
                                    source_header += f" - {' | '.join(structure_info)}"
                                
                                st.markdown(source_header)
                                
                                # Show search strategy if available
                                if 'search_strategy' in result:
                                    strategy = result['search_strategy']
                                    st.caption(f"Query Type: {strategy['query_type']} | Weights: {strategy['weights_used']}")
                                
                                # Show score breakdown if available
                                if 'score_breakdown' in result:
                                    breakdown = result['score_breakdown']
                                    breakdown_text = " | ".join([f"{k}: {v:.2f}" for k, v in breakdown.items() if v > 0])
                                    st.caption(f"Score Breakdown: {breakdown_text}")
                                
                                # Show text content
                                text_preview = result['text'][:400] + "..." if len(result['text']) > 400 else result['text']
                                st.markdown(f"```\n{text_preview}\n```")
                                
                                # Show key terms if available
                                if metadata.get('key_terms'):
                                    st.caption(f"Key Terms: {', '.join(metadata['key_terms'][:5])}")
                                if metadata.get('is_clause_variant') and metadata.get('clause_label'):
                                    st.caption(f"Clause Focus: ({metadata['clause_label']})")

                                st.divider()
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    response = error_msg
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
