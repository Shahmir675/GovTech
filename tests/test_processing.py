"""
Unit tests for processing.py (NER + Claim Extraction)
"""

import pytest
from processing import NERExtractor, ClaimExtractor, DocumentProcessor


class TestNERExtractor:
    """Test NER extraction functionality"""

    def test_extract_entities_basic(self):
        """Test basic entity extraction"""
        extractor = NERExtractor()
        text = "The District Government of Peshawar issued a notification on 15th March 2024."

        entities = extractor.extract_entities(text)

        # Should extract at least locations and dates
        assert isinstance(entities, dict)
        assert len(entities) > 0

    def test_extract_entities_empty(self):
        """Test entity extraction with empty text"""
        extractor = NERExtractor()
        entities = extractor.extract_entities("")

        assert isinstance(entities, dict)
        assert len(entities) == 0

    def test_extract_parties(self):
        """Test party extraction"""
        extractor = NERExtractor()
        text = "The petitioner District Nazim filed a case against the respondent District Government."

        parties = extractor.extract_parties(text)

        assert 'petitioners' in parties
        assert 'respondents' in parties
        assert isinstance(parties['petitioners'], list)
        assert isinstance(parties['respondents'], list)


class TestClaimExtractor:
    """Test claim extraction functionality"""

    def test_extract_claims(self, sample_narrative):
        """Test claim extraction from narrative"""
        extractor = ClaimExtractor()
        claims = extractor.extract_claims(sample_narrative)

        assert isinstance(claims, list)
        assert len(claims) > 0
        assert all('type' in claim for claim in claims)
        assert all('text' in claim for claim in claims)

    def test_extract_chronology(self, sample_narrative):
        """Test chronology extraction"""
        extractor = ClaimExtractor()
        chronology = extractor.extract_chronology(sample_narrative)

        assert isinstance(chronology, list)
        # Should find at least one date
        assert len(chronology) >= 1
        if chronology:
            assert 'date' in chronology[0]
            assert 'description' in chronology[0]

    def test_identify_inconsistencies(self, sample_narrative, sample_petition):
        """Test inconsistency detection"""
        extractor = ClaimExtractor()
        inconsistencies = extractor.identify_inconsistencies(sample_narrative, sample_petition)

        assert isinstance(inconsistencies, list)
        # May or may not find inconsistencies
        if inconsistencies:
            assert all('type' in inc for inc in inconsistencies)


class TestDocumentProcessor:
    """Test complete document processing"""

    def test_process_narrative(self, sample_narrative):
        """Test narrative processing"""
        processor = DocumentProcessor()
        result = processor.process_narrative(sample_narrative)

        assert 'entities' in result
        assert 'parties' in result
        assert 'claims' in result
        assert 'chronology' in result
        assert 'demands' in result
        assert 'processed_at' in result

    def test_process_petition(self, sample_petition):
        """Test petition processing"""
        processor = DocumentProcessor()
        result = processor.process_petition(sample_petition)

        assert 'entities' in result
        assert 'parties' in result
        assert 'claims' in result
        assert 'chronology' in result
        assert 'demands' in result

    def test_process_case(self, sample_narrative, sample_petition):
        """Test complete case processing"""
        processor = DocumentProcessor()
        result = processor.process_case(sample_narrative, sample_petition)

        assert 'narrative' in result
        assert 'petition' in result
        assert 'inconsistencies' in result
        assert 'analysis' in result
        assert result['analysis']['narrative_claim_count'] >= 0
        assert result['analysis']['petition_claim_count'] >= 0
        assert 'quality_flag_count' in result['analysis']
        assert 'section_accuracy_score' in result['analysis']
        assert 'wrong_domain_rate' in result['analysis']
        assert 'hallucination_rate' in result['analysis']

    def test_process_case_quality_checks(self, sample_narrative, sample_petition):
        """Quality checks should cover statutory accuracy and remedies"""
        processor = DocumentProcessor()
        result = processor.process_case(sample_narrative, sample_petition)

        assert 'quality_checks' in result
        quality = result['quality_checks']

        assert 'metrics' in quality
        assert 'summary' in quality

        metrics = quality['metrics']
        expected_keys = {
            'section_router',
            'statutory_accuracy',
            'grounding_principles',
            'pruning_irrelevant_law',
            'procedural_remedies',
            'evidence_trail',
            'constitutional_framing',
            'narrative_discipline'
        }

        for key in expected_keys:
            assert key in metrics
            assert 'status' in metrics[key]
            assert 'summary' in metrics[key]

        assert metrics['section_router']['status'] == 'fail'
        assert metrics['statutory_accuracy']['status'] == 'pass'
        assert metrics['procedural_remedies']['status'] == 'fail'
        assert metrics['evidence_trail']['status'] in {'warn', 'fail'}
        assert 'weighted_score_percent' in quality['summary']
        assert 'metrics_dashboard' in quality['summary']
        assert quality['summary']['router_overall_status'] == 'fail'
        assert 'router' in quality
        assert quality['router']['actions'], "Router should identify the mis-cited removal action"
        assert quality['summary']['evidence_callouts'], "Missing evidence should trigger callouts"
