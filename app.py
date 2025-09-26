import streamlit as st
import os
from typing import List
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from gemini_client import GeminiClient

# Page configuration
st.set_page_config(
    page_title="KP Local Government Act RAGBot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    .sidebar-info {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'gemini_client' not in st.session_state:
    st.session_state.gemini_client = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

def initialize_clients():
    """Initialize the vector store and Gemini client"""
    try:
        if st.session_state.vector_store is None:
            st.session_state.vector_store = VectorStore()
        if st.session_state.gemini_client is None:
            st.session_state.gemini_client = GeminiClient()
        return True
    except Exception as e:
        st.error(f"Error initializing clients: {e}")
        return False

def load_and_process_pdf():
    """Load and process the PDF document"""
    pdf_path = "THE_KHYBER_PAKHTUNKHWA_LOCAL_GOVERNMENT_ACT_2013.pdf"
    
    if not os.path.exists(pdf_path):
        st.error(f"PDF file not found: {pdf_path}")
        return False
    
    try:
        with st.spinner("Processing PDF document..."):
            # Process PDF
            processor = PDFProcessor(pdf_path)
            chunks = processor.process_pdf(chunk_size=1000, overlap=200)
            
            if not chunks:
                st.error("No text could be extracted from the PDF")
                return False
            
            # Create collection and add documents
            st.session_state.vector_store.create_collection()
            st.session_state.vector_store.add_documents(chunks)
            
            st.success(f"Successfully processed {len(chunks)} document chunks!")
            st.session_state.documents_loaded = True
            return True
            
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return False

def search_and_respond(user_query: str) -> str:
    """Search for relevant documents and generate response"""
    try:
        # Search for relevant documents
        search_results = st.session_state.vector_store.search(user_query, limit=3)
        
        if not search_results:
            return "I couldn't find relevant information in the document. Please try rephrasing your question."
        
        # Extract context documents
        context_docs = [result["text"] for result in search_results]
        
        # Generate response using Gemini
        response = st.session_state.gemini_client.generate_response(user_query, context_docs)
        
        return response
        
    except Exception as e:
        return f"Error generating response: {e}"

# Main app
def main():
    st.markdown('<h1 class="main-header">⚖️ KP Local Government Act RAGBot</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
        st.markdown("### 📋 About")
        st.markdown("This RAGBot helps you explore and understand the **Khyber Pakhtunkhwa Local Government Act 2013**.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("### 🔧 Configuration")
        
        # Check environment variables
        env_status = {
            "Qdrant URL": os.getenv('QDRANT_URL', 'Not set'),
            "Qdrant API Key": "Set" if os.getenv('QDRANT_API_KEY') else "Not set",
            "Gemini API Key": "Set" if os.getenv('GEMINI_API_KEY') else "Not set"
        }
        
        for key, value in env_status.items():
            if value == "Not set":
                st.error(f"{key}: {value}")
            else:
                st.success(f"{key}: {value if key == 'Qdrant URL' else 'Set'}")
        
        st.markdown("### 📚 Document Status")
        if st.session_state.documents_loaded:
            st.success("✅ Documents loaded")
            if st.session_state.vector_store:
                try:
                    info = st.session_state.vector_store.get_collection_info()
                    if info:
                        st.info(f"Collection: {info.config.params.vectors.size} dimensions")
                except:
                    pass
        else:
            st.warning("⏳ Documents not loaded")
        
        # Initialize and load documents button
        if st.button("🚀 Initialize RAGBot", type="primary"):
            if initialize_clients():
                load_and_process_pdf()
        
        st.markdown("### 💡 Sample Questions")
        sample_questions = [
            "What is the purpose of this Act?",
            "What are the functions of local governments?",
            "How are local councils constituted?",
            "What are the powers of the District Government?",
            "What is the role of the Provincial Government?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{question}"):
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()

    # Main content area
    if not st.session_state.documents_loaded:
        st.info("👈 Please initialize the RAGBot using the sidebar to get started!")
        st.markdown("### 🔍 What you can ask:")
        st.markdown("""
        - **General questions** about the KP Local Government Act 2013
        - **Specific provisions** and their explanations
        - **Legal definitions** and terminology
        - **Procedural requirements** and processes
        - **Powers and responsibilities** of different entities
        """)
        return
    
    # Chat interface
    st.markdown("### 💬 Ask about the KP Local Government Act 2013")
    
    # Display chat messages
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {content}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message"><strong>RAGBot:</strong> {content}</div>', 
                       unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask your question about the KP Local Government Act 2013..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate and add bot response
        with st.spinner("Thinking..."):
            response = search_and_respond(prompt)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()