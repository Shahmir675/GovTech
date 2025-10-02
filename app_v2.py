"""
RAGBot-v2: Multi-Agent Legal Assistant
Streamlit UI for narrative + petition analysis workflow
"""

import streamlit as st
import os
from dotenv import load_dotenv
from datetime import datetime

# Import orchestrator and services
from orchestrator import AgentOrchestrator, WorkflowState, WorkflowStatus
from vector_store import EnhancedQdrantVectorStore
from gemini_client import EnhancedGeminiClient
from pdf_processor import AdvancedPDFProcessor

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAGBot-v2: Multi-Agent Legal Assistant",
    page_icon="⚖️",
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
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 1.5em;
        color: #2b5b84;
        margin-top: 20px;
        margin-bottom: 10px;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 5px;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 4px solid #ffc107;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 4px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'mode' not in st.session_state:
        st.session_state.mode = 'v1_qa'  # Default to v1 Q&A mode
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'gemini_client' not in st.session_state:
        st.session_state.gemini_client = None
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = None
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'workflow_state' not in st.session_state:
        st.session_state.workflow_state = None
    if 'v1_messages' not in st.session_state:
        st.session_state.v1_messages = []


def setup_connections():
    """Setup connections to Qdrant and Gemini"""
    try:
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'kpk_local_govt_act_2013')

        if not all([gemini_api_key, qdrant_url, qdrant_api_key]):
            st.error("Please configure all required API keys in the .env file")
            return False

        # Initialize clients if not already done
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

        # Initialize orchestrator
        if st.session_state.orchestrator is None:
            st.session_state.orchestrator = AgentOrchestrator(
                vector_store=st.session_state.vector_store,
                gemini_client=st.session_state.gemini_client
            )

        return True
    except Exception as e:
        st.error(f"Error setting up connections: {str(e)}")
        return False


def load_documents():
    """Load and process the PDF document for law corpus"""
    if st.session_state.documents_loaded:
        return True

    try:
        pdf_path = "THE_KHYBER_PAKHTUNKHWA_LOCAL_GOVERNMENT_ACT_2013.pdf"

        if not os.path.exists(pdf_path):
            st.error(f"PDF file not found: {pdf_path}")
            return False

        with st.spinner("Processing PDF document (law corpus)..."):
            processor = AdvancedPDFProcessor()
            chunks_with_metadata, structure = processor.process_pdf_section_scoped(pdf_path)

            # Add documents to vector store
            st.session_state.vector_store.add_documents_with_metadata(chunks_with_metadata)
            st.session_state.documents_loaded = True

            st.success(f"Successfully processed {len(chunks_with_metadata)} sections/schedules")

            return True

    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return False


def display_v1_mode():
    """Display v1 Q&A mode (backward compatible)"""
    st.markdown('<div class="section-header">🔍 Law Q&A Mode (v1)</div>', unsafe_allow_html=True)
    st.info("Ask questions about the KPK Local Government Act 2013")

    # Display chat history
    for message in st.session_state.v1_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about the KPK Local Government Act 2013..."):
        st.session_state.v1_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    search_results = st.session_state.vector_store.smart_search(prompt, limit=5)

                    if not search_results:
                        response = "I couldn't find relevant information. Please try rephrasing."
                    else:
                        response = st.session_state.gemini_client.generate_response(prompt, search_results)

                    st.markdown(response)
                    st.session_state.v1_messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)


def display_v2_mode():
    """Display v2 Multi-Agent mode"""
    st.markdown('<div class="section-header">⚖️ Multi-Agent Legal Analysis (v2)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <strong>How it works:</strong> Provide your narrative and the opponent's petition.
    The multi-agent system will:
    <ol>
    <li>Extract entities and claims (NER + Claim Extraction)</li>
    <li>Analyze legal issues (Case Agent)</li>
    <li>Retrieve relevant law sections (Law Agent)</li>
    <li>Generate legal commentary (Drafting Agent)</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    # Input Section
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📝 Your Narrative")
        narrative = st.text_area(
            "Describe your case",
            height=300,
            placeholder="Enter your narrative here...\n\nExample: On 15th March 2024, the District Government...",
            help="Provide a detailed account of your case including dates, parties, and key events"
        )

    with col2:
        st.markdown("### 📄 Opponent's Petition")
        petition = st.text_area(
            "Paste the opponent's petition",
            height=300,
            placeholder="Enter the opponent's petition here...",
            help="Paste the text of the opponent's petition or complaint"
        )

    # Analysis Button
    if st.button("🚀 Run Multi-Agent Analysis", type="primary", disabled=not (narrative and petition)):
        with st.spinner("Running multi-agent workflow... This may take a minute."):
            try:
                # Execute workflow
                workflow_state = st.session_state.orchestrator.execute_workflow(
                    narrative=narrative,
                    petition=petition,
                    enable_fallback=True
                )

                st.session_state.workflow_state = workflow_state

                if workflow_state.status == WorkflowStatus.COMPLETED:
                    st.success("✅ Analysis complete!")
                else:
                    st.warning(f"⚠️ Analysis completed with status: {workflow_state.status.value}")

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                return

    # Display Results
    if st.session_state.workflow_state and st.session_state.workflow_state.status == WorkflowStatus.COMPLETED:
        display_analysis_results(st.session_state.workflow_state)


def display_analysis_results(state: WorkflowState):
    """Display complete analysis results in tabs"""
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)

    # Create tabs for different outputs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Executive Summary",
        "🔍 Entities & Claims",
        "⚖️ Legal Issues",
        "📚 Relevant Laws",
        "✍️ Legal Commentary"
    ])

    with tab1:
        display_executive_summary(state)

    with tab2:
        display_entities_and_claims(state)

    with tab3:
        display_legal_issues(state)

    with tab4:
        display_relevant_laws(state)

    with tab5:
        display_commentary(state)

    # Export options
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col2:
        drafting_agent = getattr(state.orchestrator, 'drafting_agent', None) if hasattr(state, 'orchestrator') else None
        if state.commentary and drafting_agent:
            report_name = f"legal_commentary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            markdown = drafting_agent.build_commentary_markdown(state.commentary)
            st.download_button(
                label="💾 Download Report",
                data=markdown.encode('utf-8'),
                file_name=report_name,
                mime="text/markdown"
            )
        else:
            st.caption("Download available after generating commentary")


def display_executive_summary(state: WorkflowState):
    """Display executive summary"""
    if state.commentary:
        st.markdown(state.commentary['executive_summary'])

    # Display workflow summary
    summary = state.orchestrator.get_summary(state) if hasattr(state, 'orchestrator') else {}

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Issues Identified", summary.get('issues_identified', 0))
    with col2:
        st.metric("Law Sections Retrieved", summary.get('law_sections_retrieved', 0))
    with col3:
        st.metric("Strengths", summary.get('strengths', 0))
    with col4:
        st.metric("Weaknesses", summary.get('weaknesses', 0))


def display_entities_and_claims(state: WorkflowState):
    """Display entities and claims extraction"""
    if not state.processed_data:
        st.warning("No processed data available")
        return

    narrative_data = state.processed_data['narrative']
    petition_data = state.processed_data['petition']

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Your Narrative")

        # Entities
        st.markdown("**Entities Extracted:**")
        for entity_type, entities in narrative_data['entities'].items():
            if entities:
                st.markdown(f"- **{entity_type}**: {', '.join([e['text'] for e in entities[:5]])}")

        # Claims
        st.markdown("**Claims:**")
        for i, claim in enumerate(narrative_data['claims'][:5], 1):
            st.markdown(f"{i}. [{claim['type']}] {claim['text'][:100]}...")

    with col2:
        st.markdown("### Opponent's Petition")

        # Entities
        st.markdown("**Entities Extracted:**")
        for entity_type, entities in petition_data['entities'].items():
            if entities:
                st.markdown(f"- **{entity_type}**: {', '.join([e['text'] for e in entities[:5]])}")

        # Claims
        st.markdown("**Claims:**")
        for i, claim in enumerate(petition_data['claims'][:5], 1):
            st.markdown(f"{i}. [{claim['type']}] {claim['text'][:100]}...")

    # Inconsistencies
    if state.processed_data['inconsistencies']:
        st.markdown("---")
        st.markdown("### ⚠️ Identified Inconsistencies")
        for inc in state.processed_data['inconsistencies']:
            severity_color = {"high": "error", "medium": "warning", "low": "info"}
            st_method = getattr(st, severity_color.get(inc.get('severity', 'low'), 'info'))
            st_method(f"**{inc['type']}**: {inc['description']}")


def display_legal_issues(state: WorkflowState):
    """Display legal issues"""
    if not state.case_analysis:
        st.warning("No case analysis available")
        return

    issues = state.case_analysis['legal_issues']
    strengths = state.case_analysis.get('strengths', [])
    weaknesses = state.case_analysis.get('weaknesses', [])
    recommendations = state.case_analysis.get('recommendations', [])

    # Issues by category
    st.markdown("### Legal Issues Identified")

    issue_categories = {}
    for issue in issues:
        category = issue.get('category', 'general')
        if category not in issue_categories:
            issue_categories[category] = []
        issue_categories[category].append(issue)

    for category, cat_issues in issue_categories.items():
        with st.expander(f"📌 {category.replace('_', ' ').title()} ({len(cat_issues)} issues)"):
            for i, issue in enumerate(cat_issues, 1):
                severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                emoji = severity_emoji.get(issue.get('severity', 'low'), '⚪')
                st.markdown(f"{emoji} **Issue {i}**: {issue['description']}")

    # Strengths and Weaknesses
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✅ Strengths")
        for strength in strengths:
            st.success(f"**{strength['category']}**: {strength['description']}")

    with col2:
        st.markdown("### ⚠️ Weaknesses")
        for weakness in weaknesses:
            severity_method = st.error if weakness.get('severity') == 'high' else st.warning
            severity_method(f"**{weakness['category']}**: {weakness['description']}")

    # Recommendations
    st.markdown("### 💡 Recommendations")
    for rec in recommendations[:5]:
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        emoji = priority_emoji.get(rec.get('priority', 'low'), '⚪')
        st.markdown(f"{emoji} **{rec['action']}**")
        st.markdown(f"   - {rec.get('details', '')}")
        st.markdown(f"   - *Rationale*: {rec.get('rationale', '')}")


def display_relevant_laws(state: WorkflowState):
    """Display relevant law sections"""
    if not state.law_retrieval:
        st.warning("No law retrieval data available")
        return

    sections = state.law_retrieval['all_relevant_sections']

    st.markdown(f"### Retrieved {len(sections)} Relevant Statutory Provisions")

    for i, section in enumerate(sections, 1):
        with st.expander(f"📖 {i}. {section['citation']} (Relevance: {section['score']:.3f})"):
            st.markdown(f"**Citation**: {section['citation']}")
            st.markdown(f"**Relevance Score**: {section['score']:.3f}")

            # Show pinpoint snippet if available
            text_to_show = section.get('sliced_text') or section.get('text', '')
            st.markdown("**Text:**")
            st.code(text_to_show[:500] + ("..." if len(text_to_show) > 500 else ""), language='text')

            # Metadata
            if section.get('metadata'):
                st.caption(f"Section: {section['metadata'].get('section_number', 'N/A')}")


def display_commentary(state: WorkflowState):
    """Display legal commentary"""
    if not state.commentary:
        st.warning("No commentary available")
        return

    commentary = state.commentary

    # Petition Critique
    st.markdown("### 📝 Petition Critique")
    st.markdown(commentary['petition_critique']['text'])

    st.markdown("---")

    # Counter-Arguments
    st.markdown("### 🎯 Counter-Arguments")
    st.markdown(commentary['counter_arguments']['text'])

    st.markdown("---")

    # Recommendations
    st.markdown("### 💡 Strategic Recommendations")
    st.markdown(commentary['recommendations']['strategic_recommendations'])

    st.markdown("---")

    # Procedural Guidance
    st.markdown("### 📋 Procedural Guidance")
    guidance = commentary['procedural_guidance']
    for step in guidance.get('steps', []):
        st.markdown(f"**Step {step['step']}**: {step['action']}")
        st.caption(f"Statutory Basis: {step.get('statutory_basis', 'N/A')}")

    st.markdown("---")

    # Disclaimer
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown(commentary['disclaimer'])
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">⚖️ RAGBot-v2: Multi-Agent Legal Assistant</h1>', unsafe_allow_html=True)
    st.caption("Powered by KPK Local Government Act 2013 | AI-Generated Legal Analysis")

    # Sidebar
    with st.sidebar:
        st.markdown("## 🛠️ Configuration")

        # Mode selector
        mode = st.radio(
            "Select Mode",
            options=['v1_qa', 'v2_multiagent'],
            format_func=lambda x: "📖 Law Q&A (v1)" if x == 'v1_qa' else "⚖️ Multi-Agent Analysis (v2)",
            key='mode'
        )

        st.markdown("---")

        # Setup connections
        if st.button("🔌 Connect to Services", type="primary"):
            if setup_connections():
                st.success("✅ Connected successfully!")
            else:
                st.error("❌ Connection failed!")

        # Load documents
        if st.session_state.vector_store is not None and not st.session_state.documents_loaded:
            if st.button("📄 Load Law Corpus", type="primary"):
                load_documents()

        # Status
        st.markdown("---")
        st.markdown("### 📊 System Status")
        st.markdown(f"**Vector Store**: {'✅ Connected' if st.session_state.vector_store else '❌ Not connected'}")
        st.markdown(f"**Gemini Client**: {'✅ Ready' if st.session_state.gemini_client else '❌ Not ready'}")
        st.markdown(f"**Law Corpus**: {'✅ Loaded' if st.session_state.documents_loaded else '❌ Not loaded'}")

        if st.session_state.vector_store and st.session_state.documents_loaded:
            info = st.session_state.vector_store.get_collection_info()
            if 'error' not in info:
                st.caption(f"Documents in corpus: {info.get('points_count', 0)}")

    # Main content area
    if st.session_state.vector_store is None or st.session_state.gemini_client is None:
        st.warning("⚠️ Please connect to services first using the sidebar.")
        return

    if not st.session_state.documents_loaded:
        st.warning("⚠️ Please load the law corpus first using the sidebar.")
        return

    # Display appropriate mode
    if st.session_state.mode == 'v1_qa':
        display_v1_mode()
    else:
        display_v2_mode()


if __name__ == "__main__":
    main()
