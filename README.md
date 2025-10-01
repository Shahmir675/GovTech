# RAGBot-v2: Multi-Agent Legal Assistant

An intelligent multi-agent legal assistant for analyzing legal disputes under the Khyber Pakhtunkhwa Local Government Act 2013. Evolved from a simple Q&A system (v1) to a comprehensive legal analysis platform (v2).

## Features

### 🆕 RAGBot-v2 (Multi-Agent Mode)
- ⚖️ **Multi-Agent Legal Analysis**: Narrative + Petition → Comprehensive Legal Commentary
- 🔍 **Processing Layer**: Named Entity Recognition (NER) + Claim Extraction
- 🧠 **Case Agent**: Extracts legal issues, identifies strengths/weaknesses, generates recommendations
- 📚 **Law Agent**: Retrieves relevant statutory provisions with hybrid search
- ✍️ **Drafting Agent**: Generates petition critique, counter-arguments, and strategic guidance
- 🎯 **Orchestration**: Automated workflow with state persistence and audit trail
- 📊 **Structured Output**: Executive summary, entities, legal issues, law sections, commentary
- 💾 **Export**: Download legal commentary reports in Markdown format

### 📖 RAGBot-v1 (Q&A Mode) - Backward Compatible
- 🤖 Intelligent Q&A about KPK Local Government Act 2013
- 🔍 Vector-based semantic search using Qdrant Cloud
- 🧠 AI-powered responses using Google Gemini
- 💬 Interactive chat interface
- 🎯 Pinpoint citations with surgical grounding

## Setup Instructions

### 1. Clone and Navigate
```bash
git clone <your-repo-url>
cd ragbot-v2
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt

# Download spaCy language model for NER
python -m spacy download en_core_web_sm
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and update with your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your actual values:

```env
# Gemini API Configuration
GEMINI_API_KEY=your_actual_gemini_api_key

# Qdrant Cloud Configuration
QDRANT_URL=your_actual_qdrant_cloud_url
QDRANT_API_KEY=your_actual_qdrant_api_key
QDRANT_COLLECTION_NAME=kpk_local_govt_act_2013

# Embedding Configuration
EMBEDDING_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
RAGBOT_MAX_CITATIONS=4

# NLP Configuration
SPACY_MODEL=en_core_web_sm
```

### 4. Get API Keys

#### Gemini API Key (for AI-powered analysis):
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

#### Qdrant Cloud (for law corpus storage):
1. Sign up at [Qdrant Cloud](https://cloud.qdrant.io/)
2. Create a new cluster (Free tier available)
3. Get your cluster URL and API key from the console
4. Add them to your `.env` file

### 5. Run the Application

#### For v2 Multi-Agent Mode:
```bash
streamlit run app_v2.py
```

#### For v1 Q&A Mode (Backward Compatible):
```bash
streamlit run app.py
```

## Usage

### Initial Setup (Both Modes)
1. **Connect to Services**: Click "🔌 Connect to Services" in the sidebar
2. **Load Law Corpus**: Click "📄 Load Law Corpus" to process the KPK Local Government Act 2013 PDF

### v1 Q&A Mode (Simple Law Queries)
1. Select "📖 Law Q&A (v1)" mode in the sidebar
2. Type your question in the chat input
3. Get AI-powered answers with statutory citations

**Sample Questions:**
- "What is the purpose of this Act?"
- "What are the powers of District Government?"
- "How are Union Councils constituted?"
- "Section 55 powers and functions"
- "Procedure for budget approval"

### v2 Multi-Agent Mode (Case Analysis)
1. Select "⚖️ Multi-Agent Analysis (v2)" mode in the sidebar
2. **Enter Your Narrative**: Describe your case with dates, parties, and events
3. **Enter Opponent's Petition**: Paste the opponent's petition text
4. Click "🚀 Run Multi-Agent Analysis"
5. Review results in tabbed interface:
   - **Executive Summary**: Key metrics and findings
   - **Entities & Claims**: Extracted information from both documents
   - **Legal Issues**: Identified issues, strengths, weaknesses
   - **Relevant Laws**: Retrieved statutory provisions with citations
   - **Legal Commentary**: Petition critique, counter-arguments, recommendations

**Example Narrative:**
```
On 15th March 2024, the District Government issued a notification removing
me from office as District Nazim without following proper procedure under
Section 55. No show-cause notice was issued, and no hearing was provided.
I seek restoration to office under Section 66.
```

**Example Petition:**
```
The District Government submits that the District Nazim was removed under
Section 55(1)(c) for gross misconduct discovered during an audit in February 2024.
Proper notice was issued on 1st March 2024 and a hearing was conducted on 10th March 2024.
```

## File Structure

```
ragbot-v2/
├── app.py                          # v1 Streamlit application (Q&A mode)
├── app_v2.py                       # v2 Streamlit application (Multi-Agent mode)
├── orchestrator.py                 # Agent orchestration and workflow management
├── processing.py                   # NER + Claim extraction (NEW)
├── case_agent.py                   # Legal issue extraction agent (NEW)
├── law_agent.py                    # Law retrieval agent (NEW)
├── drafting_agent.py               # Legal commentary generation agent (NEW)
├── pdf_processor.py                # PDF processing with section-scoped chunking
├── vector_store.py                 # Qdrant vector store with hybrid search
├── hybrid_search.py                # BM25 + semantic + TF-IDF + entity search
├── gemini_client.py                # Google Gemini API client with pinpoint citations
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (configure this!)
├── .env.example                   # Environment variable template (NEW)
├── README.md                      # This file
├── docs/
│   └── architecture.md            # Architecture documentation with diagrams (NEW)
├── tests/
│   ├── conftest.py                # Pytest fixtures (NEW)
│   ├── test_processing.py         # Processing layer tests (NEW)
│   └── test_orchestrator.py       # Orchestrator tests (NEW)
└── THE_KHYBER_PAKHTUNKHWA_LOCAL_GOVERNMENT_ACT_2013.pdf
```

## Technical Details

### v1 Architecture
- **PDF Processing**: Section-scoped chunking for complete statutory provisions
- **Vector Storage**: Qdrant Cloud with hybrid embeddings (legal-BERT + semantic models)
- **Search Engine**: Hybrid (BM25 + semantic + TF-IDF + entity matching) with cross-encoder re-ranking
- **Embeddings**: Legal-BERT ensemble for domain-specific understanding
- **AI Response**: Google Gemini 2.0 Flash with pinpoint citations
- **Chat Interface**: Streamlit with custom CSS

### v2 Architecture (Multi-Agent)
- **Processing Layer**: spaCy NER + regex-based claim extraction
- **Agent Framework**: Modular agents (Case, Law, Drafting) with orchestration
- **Case Agent**: Issue extraction with category classification and strategic analysis
- **Law Agent**: Retrieves relevant statutes using hybrid search (reuses v1 infrastructure)
- **Drafting Agent**: Gemini-powered legal commentary generation with formal templates
- **Orchestration**: Sequential workflow with state persistence and error handling
- **Workflow State**: JSON-serializable state for audit trail and resume capability
- **UI**: Tabbed interface with executive summary, entities, issues, laws, and commentary

## Precision Mode (Cleaner, Surgical Citations)

- Retrieval applies semantic + keyword thresholds to discard loosely related passages.
- Cross-encoder re-ranking (if available) prioritizes contextually tight matches.
- Results are de-duplicated per section/schedule and trimmed to relevant snippets.
- Max citations per answer are capped (default 4) to prevent citation-stuffing.
- Pinpoint citations (e.g., Section 55(1)(c)) are generated when subsection cues are detected.
- If the Act lacks an explicit mechanism asked by the user (e.g., independent arbitration), the answer states this absence clearly.

Configure citation cap via environment variable:

```
RAGBOT_MAX_CITATIONS=4
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_processing.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Troubleshooting

### Common Issues

1. **API Key Issues**:
   - Ensure all API keys are correctly set in `.env`
   - Use `.env.example` as a template
   - Never commit `.env` to version control

2. **Connection Problems**:
   - Check internet connection and API key validity
   - Verify Qdrant cluster is active in Qdrant Cloud console
   - Test Gemini API key at [Google AI Studio](https://makersuite.google.com)

3. **PDF Processing Errors**:
   - Ensure `THE_KHYBER_PAKHTUNKHWA_LOCAL_GOVERNMENT_ACT_2013.pdf` exists in project root
   - Check file permissions

4. **spaCy Model Not Found**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Empty Responses**:
   - Verify the law corpus was loaded successfully
   - Check collection status in sidebar
   - Review execution log in workflow state

6. **Out of Memory (OOM)**:
   - Reduce `RAGBOT_EMBED_BATCH` in `.env` (default: 64)
   - Use CPU instead of CUDA if GPU memory is limited
   - Process smaller text chunks

## Architecture Documentation

For detailed architecture information, workflow diagrams, and component specifications, see:

📖 **[docs/architecture.md](docs/architecture.md)**

## License

This project is for educational and research purposes.
