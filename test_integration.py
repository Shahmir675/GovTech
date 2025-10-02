#!/usr/bin/env python3
"""
Comprehensive Integration Test for RAGBot-v2
Tests all components working together end-to-end
"""

import os
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv()

def test_processing_layer():
    """Test NER and claim extraction"""
    print("\n📋 Testing Processing Layer...")
    from processing import DocumentProcessor

    processor = DocumentProcessor()

    narrative = """
    On 15th March 2024, the District Government of Peshawar issued a notification
    removing the District Nazim from office under Section 55 without following proper
    procedure. No show-cause notice was issued, and no opportunity for hearing was
    provided. This action violates principles of natural justice and is ultra vires
    Section 59 of the KPK Local Government Act 2013.

    The petitioner seeks relief to restore the petitioner to office and declare the
    impugned notification void and illegal.
    """

    petition = """
    The District Government submits that the District Nazim was removed under
    Section 59 due to gross misconduct and financial irregularities discovered
    during an audit conducted in February 2024.

    The respondent alleges that proper show-cause notice was issued on 1st March 2024,
    and a personal hearing was conducted before the Local Government Commission on
    10th March 2024, before the removal order dated 15th March 2024.

    The respondent prays that the petition be dismissed as the removal was in
    accordance with law and in public interest.
    """

    result = processor.process_case(narrative, petition)

    print(f"  ✓ Narrative claims: {result['analysis']['narrative_claim_count']}")
    print(f"  ✓ Petition claims: {result['analysis']['petition_claim_count']}")
    print(f"  ✓ Inconsistencies found: {result['analysis']['inconsistency_count']}")
    print(f"  ✓ Quality checks: {len(result['quality_checks']['metrics'])} metrics evaluated")
    print(f"  ✓ Section accuracy: {result['analysis']['section_accuracy_score']}%")
    print(f"  ✓ Weighted quality score: {result['quality_checks']['summary']['weighted_score_percent']}%")

    # Validate quality checks
    assert 'quality_checks' in result
    assert 'metrics' in result['quality_checks']
    assert len(result['quality_checks']['metrics']) == 8

    print("  ✅ Processing layer passed!")
    return True


def test_vector_store():
    """Test vector store and search"""
    print("\n🔍 Testing Vector Store...")
    from vector_store import EnhancedQdrantVectorStore

    url = os.getenv('QDRANT_URL')
    api_key = os.getenv('QDRANT_API_KEY')
    collection_name = os.getenv('QDRANT_COLLECTION_NAME')

    if not all([url, api_key, collection_name]):
        print("  ⚠️  Vector store credentials not configured, skipping")
        return True

    vector_store = EnhancedQdrantVectorStore(url=url, api_key=api_key, collection_name=collection_name)

    # Test search
    results = vector_store.smart_search('Who can suspend a chairman?', limit=5)
    print(f"  ✓ Smart search returned {len(results)} results")

    if results:
        print(f"  ✓ Top result score: {results[0].get('score', 0):.3f}")
        print(f"  ✓ Section: {results[0].get('metadata', {}).get('section_number', 'N/A')}")

    print("  ✅ Vector store passed!")
    return True


def test_case_agent():
    """Test case analysis agent"""
    print("\n⚖️  Testing Case Agent...")
    from case_agent import CaseAgent
    from unittest.mock import Mock

    # Create mock Gemini client
    mock_gemini = Mock()
    mock_gemini.generate_response = Mock(
        return_value="Test analysis response"
    )

    case_agent = CaseAgent(gemini_client=mock_gemini)

    narrative = "The District Nazim was removed without hearing under Section 59."
    petition = "The removal was lawful under Section 59 after proper hearing."

    result = case_agent.analyze_case(narrative, petition)

    print(f"  ✓ Case summary generated")
    print(f"  ✓ Legal issues identified: {len(result.get('legal_issues', []))}")
    print(f"  ✓ Strengths: {len(result.get('strengths', []))}")
    print(f"  ✓ Weaknesses: {len(result.get('weaknesses', []))}")

    assert 'case_summary' in result
    assert 'legal_issues' in result

    print("  ✅ Case agent passed!")
    return True


def test_law_agent():
    """Test law retrieval agent"""
    print("\n📚 Testing Law Agent...")
    from law_agent import LawAgent
    from unittest.mock import Mock

    mock_vector_store = Mock()
    mock_vector_store.smart_search = Mock(return_value=[
        {
            'text': 'Section 59: Suspension and removal of Chairman...',
            'score': 0.85,
            'metadata': {'section_number': '59', 'title': 'Suspension and removal'},
            'citation': 'Section 59'
        }
    ])

    law_agent = LawAgent(vector_store=mock_vector_store)

    # Pass list of issues, not dict
    legal_issues = [
        {
            'id': 'issue_1',
            'description': 'Whether removal was lawful',
            'category': 'procedural'
        }
    ]

    result = law_agent.retrieve_relevant_law(legal_issues)

    print(f"  ✓ Issue-law mapping: {len(result.get('issue_law_mapping', []))}")
    print(f"  ✓ Relevant sections: {len(result.get('all_relevant_sections', []))}")

    assert 'issue_law_mapping' in result
    assert 'all_relevant_sections' in result

    print("  ✅ Law agent passed!")
    return True


def test_drafting_agent():
    """Test drafting agent"""
    print("\n✍️  Testing Drafting Agent...")
    from drafting_agent import DraftingAgent
    from unittest.mock import Mock

    mock_gemini = Mock()
    mock_gemini.generate_response = Mock(
        return_value="Test draft content"
    )

    drafting_agent = DraftingAgent(gemini_client=mock_gemini)

    case_analysis = {
        'case_summary': {'parties': {'user_party': 'District Nazim'}},
        'legal_issues': [{'description': 'procedural violation'}],
        'strengths': [{'description': 'clear statutory reference'}],
        'weaknesses': [{'description': 'limited evidence'}]
    }

    law_retrieval = {
        'all_relevant_sections': [
            {'citation': 'Section 59', 'text': 'Suspension of Chairman...'}
        ]
    }

    result = drafting_agent.generate_commentary(case_analysis, law_retrieval)

    print(f"  ✓ Commentary generated")
    print(f"  ✓ Result type: {type(result)}")

    assert result is not None

    print("  ✅ Drafting agent passed!")
    return True


def test_orchestrator():
    """Test orchestrator workflow"""
    print("\n🎯 Testing Orchestrator...")
    from orchestrator import WorkflowState

    state = WorkflowState()

    state.log_step('processing', 'Extracted entities')
    state.log_step('case_analysis', 'Analyzed legal issues')
    state.add_warning('Missing evidence')

    summary = state.to_dict()

    print(f"  ✓ Execution log: {len(summary['execution_log'])}")
    print(f"  ✓ Warnings: {len(summary['warnings'])}")

    assert len(summary['execution_log']) >= 2
    assert len(summary['warnings']) == 1

    print("  ✅ Orchestrator passed!")
    return True


def main():
    """Run all integration tests"""
    print("=" * 60)
    print("🚀 RAGBot-v2 Integration Test Suite")
    print("=" * 60)

    tests = [
        ("Processing Layer", test_processing_layer),
        ("Vector Store", test_vector_store),
        ("Case Agent", test_case_agent),
        ("Law Agent", test_law_agent),
        ("Drafting Agent", test_drafting_agent),
        ("Orchestrator", test_orchestrator),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n🎉 All integration tests passed!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
