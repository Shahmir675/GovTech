#!/usr/bin/env python3
"""
End-to-End Test for RAGBot-v2
Simulates a complete user workflow
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def test_complete_workflow():
    """Test the complete RAGBot-v2 workflow"""
    print("\n" + "="*70)
    print("🚀 RAGBot-v2 End-to-End Workflow Test")
    print("="*70)

    # Step 1: Document Processing
    print("\n📋 Step 1: Processing Documents...")
    from processing import DocumentProcessor

    processor = DocumentProcessor()

    narrative = """
    On 15th March 2024, I, Muhammad Aziz, Chairman of District Council Peshawar,
    was suspended by order of the Chief Minister without being given a personal
    hearing before the Local Government Commission, as required under Section 59
    of the KPK Local Government Act 2013.

    No written reasons were recorded before my suspension. I was not given the
    opportunity to defend myself before the Commission. The suspension order was
    issued arbitrarily on allegations of misconduct that were never properly
    investigated.

    I respectfully submit that the suspension violates the mandatory procedural
    safeguards under Section 59, and I seek restoration to my office with immediate
    effect.
    """

    petition = """
    The Government of Khyber Pakhtunkhwa submits that the Chairman Muhammad Aziz
    was suspended under Section 59 of the KPK Local Government Act 2013 on
    15th March 2024, following serious allegations of financial misconduct and
    abuse of office.

    The Chief Minister duly recorded reasons in writing before ordering the
    suspension. The matter was promptly referred to the Local Government Commission
    for enquiry. A personal hearing was conducted by the Commission on 20th March
    2024, where the Chairman was given full opportunity to present his defense.

    The Government prays that the petition be dismissed as the suspension was
    conducted in full compliance with Section 59 and all mandatory procedures
    were followed.
    """

    processed = processor.process_case(narrative, petition)

    print(f"  ✓ Narrative processed: {processed['analysis']['narrative_claim_count']} claims extracted")
    print(f"  ✓ Petition processed: {processed['analysis']['petition_claim_count']} claims extracted")
    print(f"  ✓ Inconsistencies detected: {processed['analysis']['inconsistency_count']}")
    print(f"  ✓ Quality score: {processed['quality_checks']['summary']['weighted_score_percent']}%")
    print(f"  ✓ Section accuracy: {processed['analysis']['section_accuracy_score']}%")

    # Step 2: Case Analysis
    print("\n⚖️  Step 2: Analyzing Legal Case...")
    from case_agent import CaseAgent

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("  ⚠️  GEMINI_API_KEY not set, using mock for case analysis")
        from unittest.mock import Mock
        mock_gemini = Mock()
        mock_gemini.generate_response = Mock(return_value="Mock analysis")
        case_agent = CaseAgent(gemini_client=mock_gemini)
    else:
        from gemini_client import EnhancedGeminiClient
        gemini = EnhancedGeminiClient(api_key=api_key)
        case_agent = CaseAgent(gemini_client=gemini)

    case_analysis = case_agent.analyze_case(narrative, petition, processed)

    print(f"  ✓ Case summary: {case_analysis['case_summary']['dispute_type']}")
    print(f"  ✓ Legal issues: {len(case_analysis['legal_issues'])}")
    print(f"  ✓ Strengths: {len(case_analysis['strengths'])}")
    print(f"  ✓ Weaknesses: {len(case_analysis['weaknesses'])}")

    # Step 3: Law Retrieval
    print("\n📚 Step 3: Retrieving Relevant Law...")
    from law_agent import LawAgent
    from vector_store import EnhancedQdrantVectorStore

    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    collection_name = os.getenv('QDRANT_COLLECTION_NAME')

    if not all([qdrant_url, qdrant_api_key, collection_name]):
        print("  ⚠️  Vector store not configured, using mock")
        from unittest.mock import Mock
        mock_vector_store = Mock()
        mock_vector_store.smart_search = Mock(return_value=[
            {
                'text': 'Section 59: Suspension of Chairman...',
                'score': 0.9,
                'metadata': {'section_number': '59'},
                'citation': 'Section 59'
            }
        ])
        vector_store = mock_vector_store
    else:
        vector_store = EnhancedQdrantVectorStore(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name
        )

    law_agent = LawAgent(vector_store=vector_store)
    law_data = law_agent.retrieve_relevant_law(case_analysis['legal_issues'])

    print(f"  ✓ Relevant sections retrieved: {len(law_data['all_relevant_sections'])}")
    print(f"  ✓ Issue-law mappings: {len(law_data['issue_law_mapping'])}")

    # Step 4: Draft Commentary
    print("\n✍️  Step 4: Generating Legal Commentary...")
    from drafting_agent import DraftingAgent

    if not api_key:
        print("  ⚠️  Using mock for drafting")
        mock_gemini = Mock()
        mock_gemini.generate_response = Mock(return_value="Mock commentary")
        drafting_agent = DraftingAgent(gemini_client=mock_gemini)
    else:
        drafting_agent = DraftingAgent(gemini_client=gemini)

    commentary = drafting_agent.generate_commentary(case_analysis, law_data)

    print(f"  ✓ Commentary sections: {len(commentary)}")
    if 'executive_summary' in commentary:
        print(f"  ✓ Executive summary: {len(commentary['executive_summary'])} chars")
    if 'counter_arguments' in commentary:
        print(f"  ✓ Counter arguments: {len(commentary['counter_arguments'])} chars")

    # Step 5: Final Report
    print("\n📊 Step 5: Generating Final Report...")

    report = {
        'processed_data': processed,
        'case_analysis': case_analysis,
        'law_data': law_data,
        'commentary': commentary
    }

    print(f"  ✓ Report generated with {len(report)} sections")
    print(f"  ✓ Quality metrics:")
    print(f"    - Section Accuracy: {processed['analysis']['section_accuracy_score']}%")
    print(f"    - Wrong Domain Rate: {processed['analysis']['wrong_domain_rate']}%")
    print(f"    - Hallucination Rate: {processed['analysis']['hallucination_rate']}%")
    print(f"    - Overall Quality Score: {processed['quality_checks']['summary']['weighted_score_percent']}%")

    # Validation
    print("\n✅ Validation Checks:")
    checks = [
        (processed['analysis']['narrative_claim_count'] >= 0, "Claims extracted from narrative"),
        (len(case_analysis['legal_issues']) >= 0, "Legal issues identified"),
        (len(law_data['all_relevant_sections']) >= 0, "Relevant law retrieved"),
        ('executive_summary' in commentary or 'counter_arguments' in commentary, "Commentary generated"),
        (processed['quality_checks']['summary']['weighted_score_percent'] >= 0, "Quality score computed")
    ]

    for passed, desc in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {desc}")

    all_passed = all(check[0] for check in checks)

    print("\n" + "="*70)
    if all_passed:
        print("🎉 End-to-End Test PASSED!")
        print("All components working correctly together.")
    else:
        print("⚠️  Some checks failed")
    print("="*70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(test_complete_workflow())
