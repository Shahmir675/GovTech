# KP Local Government Act RAGBot

An intelligent Retrieval-Augmented Generation (RAG) chatbot for exploring and understanding the **Khyber Pakhtunkhwa Local Government Act 2013**. Built with Streamlit, Qdrant Cloud, and Google's Gemini API.

## Features

- 🔍 **Intelligent Document Search**: Uses vector similarity search to find relevant sections
- 🤖 **AI-Powered Responses**: Leverages Google Gemini for natural language understanding
- 📚 **Context-Aware**: Provides answers based on the actual document content
- 🎨 **User-Friendly Interface**: Clean Streamlit interface with chat functionality
- ⚡ **Fast and Accurate**: Efficient vector search with semantic understanding

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd ragbot-v1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Edit the `.env` file with your actual API keys:

```env
# Qdrant Cloud Configuration
QDRANT_URL=https://your-cluster-url.qdrant.tech:6333
QDRANT_API_KEY=your_qdrant_api_key_here

# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Vector Database Configuration
COLLECTION_NAME=kp_local_government_act
VECTOR_SIZE=768
```

#### Getting API Keys:

**Qdrant Cloud:**
1. Sign up at [qdrant.tech](https://qdrant.tech/)
2. Create a new cluster
3. Get your cluster URL and API key from the dashboard

**Google Gemini API:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

### 4. Run the Application
```bash
streamlit run app.py
```

## Usage

1. **Initialize the RAGBot**: Click the "🚀 Initialize RAGBot" button in the sidebar
2. **Wait for Processing**: The system will process the PDF and create vector embeddings
3. **Ask Questions**: Use the chat interface to ask questions about the Act
4. **Explore**: Try the sample questions or ask your own

## Example Questions

- "What is the purpose of this Act?"
- "What are the functions of local governments?"
- "How are local councils constituted?"
- "What are the powers of the District Government?"
- "What is the role of the Provincial Government?"

## Project Structure

```
ragbot-v1/
├── app.py                     # Main Streamlit application
├── pdf_processor.py           # PDF text extraction and chunking
├── vector_store.py           # Qdrant vector database integration
├── gemini_client.py          # Google Gemini API client
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables
├── THE_KHYBER_PAKHTUNKHWA_LOCAL_GOVERNMENT_ACT_2013.pdf
└── README.md                 # This file
```

## Technical Details

### Components

1. **PDF Processing**: Extracts text from PDF and splits into overlapping chunks
2. **Vector Store**: Uses Qdrant Cloud for storing and searching document embeddings
3. **AI Integration**: Google Gemini provides natural language understanding
4. **Web Interface**: Streamlit provides the interactive chat interface

### Architecture

```
User Query → Vector Search → Context Retrieval → Gemini API → Response
```

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure all API keys are correctly set in `.env`
2. **PDF Not Found**: Make sure the PDF file is in the project root
3. **Vector Store Issues**: Check Qdrant Cloud connection and API key
4. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`

### Environment Variables Check

The sidebar shows the status of your environment variables:
- ✅ Green: Properly configured
- ❌ Red: Missing or incorrect configuration

## Dependencies

- **streamlit**: Web interface framework
- **google-generativeai**: Google Gemini API client
- **qdrant-client**: Qdrant vector database client
- **PyPDF2**: PDF text extraction
- **sentence-transformers**: Text embedding models
- **python-dotenv**: Environment variable management

## License

This project is for educational and research purposes related to understanding local government legislation.