# KPK Local Government Act 2013 - RAGBot

An intelligent Retrieval-Augmented Generation (RAG) chatbot for querying the Khyber Pakhtunkhwa Local Government Act 2013. Built with Streamlit, Qdrant Cloud, and Google Gemini API.

## Features

- 🤖 Intelligent Q&A about KPK Local Government Act 2013
- 🔍 Vector-based semantic search using Qdrant Cloud
- 🧠 AI-powered responses using Google Gemini
- 💬 Interactive Streamlit web interface
- 📄 Automatic PDF processing and chunking
- 🎯 Source attribution for transparency

## Setup Instructions

### 1. Clone and Navigate
```bash
git clone <your-repo-url>
cd ragbot-v1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Update the `.env` file with your API keys:

```env
# Gemini API Configuration
GEMINI_API_KEY=your_actual_gemini_api_key

# Qdrant Cloud Configuration
QDRANT_URL=your_actual_qdrant_cloud_url
QDRANT_API_KEY=your_actual_qdrant_api_key

# Collection Settings
QDRANT_COLLECTION_NAME=kpk_local_govt_act_2013
```

### 4. Get API Keys

#### Gemini API Key (for chat responses):
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

#### Qdrant Cloud:
1. Sign up at [Qdrant Cloud](https://cloud.qdrant.io/)
2. Create a new cluster
3. Get your cluster URL and API key
4. Add them to your `.env` file

### 5. Run the Application
```bash
streamlit run app.py
```

## Usage

1. **Connect to Services**: Click "Connect to Services" in the sidebar
2. **Load PDF Document**: Click "Load PDF Document" to process the legal document
3. **Start Chatting**: Ask questions about the KPK Local Government Act 2013

### Sample Questions
- "What is the purpose of this Act?"
- "What are the powers of local government?"
- "How are local councils constituted?"
- "What are the functions of Union Councils?"
- "What is the role of District Government?"

## File Structure

```
ragbot-v1/
├── app.py                          # Main Streamlit application
├── pdf_processor.py                # PDF text extraction and chunking
├── vector_store.py                 # Qdrant vector store integration
├── gemini_client.py                # Google Gemini API client
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (configure this!)
├── README.md                      # This file
└── THE_KHYBER_PAKHTUNKHWA_LOCAL_GOVERNMENT_ACT_2013.pdf
```

## Technical Details

- **PDF Processing**: Uses PyPDF2 for text extraction and LangChain for intelligent chunking
- **Vector Storage**: Qdrant Cloud for scalable vector storage and similarity search
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2) for semantic understanding
- **Chat Interface**: Streamlit for responsive web UI
- **AI Response**: Google Gemini 1.5 Flash for generating contextual answers

## Troubleshooting

1. **API Key Issues**: Ensure all API keys are correctly set in `.env`
2. **Connection Problems**: Check your internet connection and API key validity
3. **PDF Processing Errors**: Ensure the PDF file exists in the project directory
4. **Empty Responses**: Verify the document was loaded successfully

## License

This project is for educational and research purposes.