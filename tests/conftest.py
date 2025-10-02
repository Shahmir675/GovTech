"""
Pytest configuration and shared fixtures for RAGBot-v2 tests
"""

import os
import sys
from unittest.mock import Mock, MagicMock

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Sample test data
SAMPLE_NARRATIVE = """
On 15th March 2024, the District Government of Peshawar issued a notification removing
the District Nazim from office without following proper procedure under Section 55 of
the KPK Local Government Act 2013. No show-cause notice was issued, and no opportunity
for hearing was provided. This action violates principles of natural justice and is
ultra vires the Act.

The petitioner seeks relief under Section 66 to restore the petitioner to office and
declare the impugned notification void and illegal.
"""

SAMPLE_PETITION = """
The District Government submits that the District Nazim was removed under Section 55(1)(c)
due to gross misconduct and financial irregularities discovered during an audit conducted
in February 2024.

The respondent alleges that proper show-cause notice was issued on 1st March 2024, and
a hearing was conducted on 10th March 2024, before the removal order dated 15th March 2024.

The respondent prays that the petition be dismissed as the removal was in accordance
with law and in public interest.
"""


@pytest.fixture
def sample_narrative():
    """Provide sample narrative text"""
    return SAMPLE_NARRATIVE


@pytest.fixture
def sample_petition():
    """Provide sample petition text"""
    return SAMPLE_PETITION


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    mock_store = Mock()
    mock_store.smart_search = Mock(return_value=[
        {
            'text': 'Section 55: Removal of Nazim or Naib Nazim...',
            'score': 0.85,
            'metadata': {
                'section_number': '55',
                'title': 'Removal of Nazim or Naib Nazim',
                'section': 'Removal of Nazim or Naib Nazim'
            },
            'citation': 'Section 55: Removal of Nazim or Naib Nazim'
        }
    ])
    return mock_store


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client for testing"""
    mock_client = Mock()
    mock_client.model = Mock()
    mock_client.model.generate_content = Mock(
        return_value=Mock(text="This is a test response")
    )
    mock_client.generate_response = Mock(
        return_value="Test response from Gemini"
    )
    return mock_client


@pytest.fixture
def sample_processed_data():
    """Provide sample processed data"""
    return {
        'narrative': {
            'entities': {
                'PERSON': [{'text': 'District Nazim', 'start': 50, 'end': 64}],
                'ORG': [{'text': 'District Government', 'start': 20, 'end': 39}],
                'DATE_PATTERN': [{'text': '15th March 2024', 'start': 3, 'end': 18}]
            },
            'parties': {
                'petitioners': ['District Nazim'],
                'respondents': [],
                'authorities': ['District Government']
            },
            'claims': [
                {
                    'type': 'violation',
                    'text': 'removing the District Nazim from office without following proper procedure',
                    'position': 60
                }
            ],
            'chronology': [
                {'date': '15th March 2024', 'description': 'District Government issued notification removing District Nazim'}
            ],
            'demands': []
        },
        'petition': {
            'entities': {
                'PERSON': [{'text': 'District Nazim', 'start': 40, 'end': 54}],
                'STATUTE_REF': [{'text': 'Section 55(1)(c)', 'start': 70, 'end': 86}]
            },
            'parties': {
                'petitioners': [],
                'respondents': ['District Nazim'],
                'authorities': ['District Government']
            },
            'claims': [
                {
                    'type': 'allegation',
                    'text': 'gross misconduct and financial irregularities',
                    'position': 90
                }
            ],
            'chronology': [
                {'date': '1st March 2024', 'description': 'show-cause notice issued'},
                {'date': '10th March 2024', 'description': 'hearing conducted'},
                {'date': '15th March 2024', 'description': 'removal order'}
            ],
            'demands': []
        },
        'inconsistencies': [
            {
                'type': 'date_mismatch',
                'description': 'Dates mentioned in narrative differ from petition',
                'severity': 'medium'
            }
        ],
        'analysis': {
            'narrative_claim_count': 1,
            'petition_claim_count': 1,
            'inconsistency_count': 1,
            'critical_inconsistencies': []
        }
    }


@pytest.fixture
def sample_case_analysis():
    """Provide sample case analysis"""
    return {
        'case_summary': {
            'parties': {
                'user_party': 'District Nazim',
                'opponent_party': 'District Government',
                'authorities_involved': ['District Government']
            },
            'issue_count': 2,
            'dispute_type': 'Administrative Law Dispute',
            'complexity': 'medium'
        },
        'legal_issues': [
            {
                'id': 'issue_1',
                'description': 'Whether removal was conducted in accordance with Section 55',
                'category': 'procedural',
                'severity': 'high'
            },
            {
                'id': 'issue_2',
                'description': 'Whether principles of natural justice were violated',
                'category': 'administrative',
                'severity': 'high'
            }
        ],
        'strengths': [
            {
                'category': 'legal_grounding',
                'description': 'Clear statutory reference to Section 55',
                'impact': 'high'
            }
        ],
        'weaknesses': [
            {
                'category': 'documentation',
                'description': 'Limited evidence of procedural violations',
                'severity': 'medium'
            }
        ],
        'recommendations': [
            {
                'priority': 'high',
                'action': 'Obtain copy of show-cause notice',
                'details': 'Verify if proper notice was given',
                'rationale': 'Critical for procedural argument'
            }
        ]
    }


@pytest.fixture
def sample_law_retrieval():
    """Provide sample law retrieval results"""
    return {
        'issue_law_mapping': [
            {
                'issue_id': 'issue_1',
                'relevant_sections': [
                    {
                        'text': 'Section 55: Removal of Nazim...',
                        'score': 0.85,
                        'citation': 'Section 55: Removal of Nazim',
                        'section_number': '55'
                    }
                ],
                'retrieval_confidence': 'high'
            }
        ],
        'all_relevant_sections': [
            {
                'text': 'Section 55: Removal of Nazim or Naib Nazim...',
                'score': 0.85,
                'metadata': {'section_number': '55'},
                'citation': 'Section 55: Removal of Nazim',
                'section_number': '55',
                'rank': 1
            }
        ],
        'metadata': {
            'total_issues': 2,
            'total_unique_sections': 1
        }
    }
