"""
Processing Layer for RAGBot-v2: NER + Claim Extraction

This module provides Named Entity Recognition (NER) and claim extraction
capabilities for processing user narratives and opponent petitions.
"""

import re
import subprocess
import shutil
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict
import json
from datetime import datetime
from textwrap import indent


def _uv_exists() -> Optional[str]:
    """Return the uv executable path if available."""
    return shutil.which("uv")


def _uv_install(target: str, uv_path: Optional[str] = None) -> bool:
    """Run `uv pip install <target>` and return True if it succeeds."""
    if uv_path is None:
        uv_path = _uv_exists()
    if not uv_path:
        return False

    print(f"⚙️  Running: uv pip install {target}")
    result = subprocess.run([uv_path, "pip", "install", target], check=False)
    if result.returncode != 0:
        print(f"❌ uv pip install {target} failed with exit code {result.returncode}")
        return False
    return True


def _resolve_model_candidates(model_name: str, spacy_version: Optional[str]) -> List[str]:
    """Return install targets (wheel URLs) compatible with the current spaCy version."""
    candidates: List[str] = []
    if spacy_version:
        major_minor = ".".join(spacy_version.split(".")[:2])
        version_map = _MODEL_VERSION_LINKS.get(model_name, {})
        wheel = version_map.get(major_minor)
        if wheel:
            candidates.append(wheel)

    default_wheel = _MODEL_LINKS.get(model_name)
    if default_wheel and default_wheel not in candidates:
        candidates.append(default_wheel)

    if model_name not in candidates:
        candidates.append(model_name)

    return candidates


_MODEL_VERSION_LINKS = {
    "en_core_web_sm": {
        "3.8": "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
        "3.7": "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl",
    }
}


_MODEL_LINKS = {
    "en_core_web_sm": "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
}


def _ensure_spacy_installed() -> bool:
    """Attempt to install spaCy via uv if it is missing."""
    return _uv_install("spacy")


def _ensure_model_installed(model_name: str, spacy_version: Optional[str]) -> bool:
    """Attempt to install the requested spaCy model via uv."""
    uv_path = _uv_exists()
    if not uv_path:
        return False

    for target in _resolve_model_candidates(model_name, spacy_version):
        if _uv_install(target, uv_path=uv_path):
            return True
    return False


# Try to import spaCy, installing it via uv if necessary
try:
    import spacy
    from spacy.language import Language
    SPACY_AVAILABLE = True
except ImportError:
    print("⚠️  spaCy not available. Attempting installation via uv…")
    if _ensure_spacy_installed():
        try:
            import spacy
            from spacy.language import Language
            SPACY_AVAILABLE = True
            print("✅ Installed spaCy with uv")
        except ImportError:
            SPACY_AVAILABLE = False
    else:
        SPACY_AVAILABLE = False

if not SPACY_AVAILABLE:
    print(
        "⚠️  spaCy is not available. Install it manually with uv, for example:\n"
        + indent(
            "\n".join(
                [
                    "uv pip install spacy",
                    "uv pip install 'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'",
                ]
            ),
            "  ",
        )
    )


# Statutory metadata for Khyber Pakhtunkhwa Local Government Act, 2013.
_KNOWN_SECTION_BASES: Set[str] = {str(i) for i in range(1, 125)}
_KNOWN_SECTION_BASES.update({'23A', '25A', '77A', '78A', '115A', '115B', '120A'})

# Domain ranges based on chapters to enforce subject-specific routing.
SECTION_DOMAIN_RANGES: Dict[str, Set[int]] = {
    'taxation': set(range(42, 50)),  # Chapter X
    'discipline': set(range(54, 66)),  # Chapter XII (Supervision / disciplinary controls)
    'elections': set(range(74, 104)),  # Chapter XIV (Local council elections)
    'appeals': {111},  # Section 111: Appeals
}


def _compile_patterns(patterns: List[str]) -> List[re.Pattern]:
    """Compile regex patterns with IGNORECASE flag."""
    return [re.compile(p, re.IGNORECASE) for p in patterns]


ACTION_ROUTER_RULES: List[Dict[str, Any]] = [
    {
        'id': 'removal_chairman',
        'description': 'Suspension or removal of a Chairman/Nazim',
        'domain': 'discipline',
        'expected_sections': ['59'],
        'regex': r'(?:suspend(?:ed|ing)?|remov(?:e|al|ing)|de-?notif(?:y|ication))[\s\S]{0,160}(?:chairman|nazim|naib\s+nazim|mayor)',
        'mandatory_steps': [
            {
                'id': 'written_reasons',
                'description': 'Chief Minister recorded reasons in writing before suspension',
                'patterns': _compile_patterns([
                    r'reasons?\s+to\s+be\s+recorded',
                    r'recorded\s+in\s+writing',
                    r'written\s+order'
                ])
            },
            {
                'id': 'commission_reference',
                'description': 'Matter referred to the Local Government Commission for enquiry',
                'patterns': _compile_patterns([
                    r'local\s+government\s+commission',
                    r'referred\s+to\s+the\s+commission'
                ])
            },
            {
                'id': 'personal_hearing',
                'description': 'Personal hearing provided to the Chairman by the Commission',
                'patterns': _compile_patterns([
                    r'personal\s+hearing',
                    r'opportunity\s+of\s+(?:being\s+)?heard',
                    r'hearing\s+was\s+conducted'
                ])
            },
            {
                'id': 'thirty_day_limit',
                'description': 'Decision concluded within thirty days of suspension',
                'patterns': _compile_patterns([
                    r'within\s+thirty\s+days',
                    r'within\s+30\s+days'
                ])
            },
        ],
        'required_evidence': [
            {
                'id': 'suspension_order',
                'description': 'Copy of the Chief Minister suspension order',
                'patterns': _compile_patterns([
                    r'suspension\s+order',
                    r'order\s+dated'
                ])
            },
            {
                'id': 'commission_notice',
                'description': 'Notice issued by the Local Government Commission',
                'patterns': _compile_patterns([
                    r'commission\s+notice',
                    r'notice\s+dated'
                ])
            },
            {
                'id': 'hearing_minutes',
                'description': 'Minutes or proceedings of the personal hearing',
                'patterns': _compile_patterns([
                    r'minutes\s+of\s+hearing',
                    r'hearing\s+proceedings',
                    r'hearing\s+record'
                ])
            },
        ],
    },
    {
        'id': 'tax_imposition',
        'description': 'Imposition or modification of a local tax or levy',
        'domain': 'taxation',
        'expected_sections': ['42'],
        'regex': r'(?:impos(?:e|ition)|levy|notification)[\s\S]{0,160}(?:tax|fee|cess)',
        'mandatory_steps': [
            {
                'id': 'public_notice',
                'description': 'Proposed tax published for public notice',
                'patterns': _compile_patterns([
                    r'public\s+notice',
                    r'published\s+in\s+the\s+gazette',
                    r'publication\s+of\s+the\s+tax'
                ])
            },
            {
                'id': 'invite_objections',
                'description': 'Public objections invited and heard',
                'patterns': _compile_patterns([
                    r'invit(?:e|ing)\s+objections?',
                    r'hearing\s+of\s+objections',
                    r'considered\s+objections'
                ])
            },
            {
                'id': 'council_approval',
                'description': 'Respective council approved the tax proposal',
                'patterns': _compile_patterns([
                    r'(?:tehsil|district|city)\s+council\s+approval',
                    r'approved\s+by\s+the\s+council',
                    r'resolution\s+of\s+the\s+council'
                ])
            },
            {
                'id': 'enforcement_date',
                'description': 'Enforcement date notified',
                'patterns': _compile_patterns([
                    r'enforcement\s+date',
                    r'shall\s+come\s+into\s+force',
                    r'effective\s+from'
                ])
            },
        ],
        'required_evidence': [
            {
                'id': 'gazette_copy',
                'description': 'Copy of Gazette/notification placing the tax in force',
                'patterns': _compile_patterns([
                    r'gazette\s+notification',
                    r'notified\s+vide'
                ])
            },
            {
                'id': 'objection_record',
                'description': 'Register or minutes showing objections were heard',
                'patterns': _compile_patterns([
                    r'objection\s+register',
                    r'minutes\s+of\s+objections',
                    r'hearing\s+minutes'
                ])
            },
            {
                'id': 'council_minutes',
                'description': 'Minutes/resolution of council approving the levy',
                'patterns': _compile_patterns([
                    r'council\s+resolution',
                    r'minutes\s+of\s+(?:tehsil|district|city)\s+council'
                ])
            },
        ],
    },
    {
        'id': 'election_petition',
        'description': 'Election petition challenging a local council election',
        'domain': 'elections',
        'expected_sections': ['87'],
        'regex': r'election\s+petition',
        'mandatory_steps': [
            {
                'id': 'candidate_only',
                'description': 'Petition filed by a contesting candidate',
                'patterns': _compile_patterns([
                    r'candidate\s+filed',
                    r'filed\s+by\s+the\s+candidate'
                ])
            },
            {
                'id': 'tribunal_constituted',
                'description': 'Election Commission appointed an Election Tribunal',
                'patterns': _compile_patterns([
                    r'election\s+tribunal',
                    r'tribunal\s+appointed',
                    r'notified\s+tribunal'
                ])
            },
            {
                'id': 'prescribed_manner',
                'description': 'Petition filed or tried in prescribed manner/rules',
                'patterns': _compile_patterns([
                    r'prescribed\s+manner',
                    r'in\s+accordance\s+with\s+the\s+rules',
                    r'procedure\s+prescribed'
                ])
            },
        ],
        'required_evidence': [
            {
                'id': 'petition_copy',
                'description': 'Certified copy of the election petition',
                'patterns': _compile_patterns([
                    r'copy\s+of\s+the\s+petition',
                    r'annexed\s+petition'
                ])
            },
            {
                'id': 'tribunal_notification',
                'description': 'Notification establishing the Election Tribunal',
                'patterns': _compile_patterns([
                    r'tribunal\s+notification',
                    r'election\s+commission\s+notification'
                ])
            },
        ],
    },
    {
        'id': 'local_government_appeal',
        'description': 'Statutory appeal against a local government order',
        'domain': 'appeals',
        'expected_sections': ['111'],
        'regex': r'appeal[\s\S]{0,160}(?:local\s+government|chairman|order)',
        'mandatory_steps': [
            {
                'id': 'appellate_authority',
                'description': 'Appeal identifies the specified appellate authority',
                'patterns': _compile_patterns([
                    r'appeal\s+before\s+the\s+[a-z\s]+authority',
                    r'appellate\s+authority'
                ])
            },
            {
                'id': 'limitation',
                'description': 'Appeal filed within prescribed limitation/period',
                'patterns': _compile_patterns([
                    r'within\s+the\s+prescribed\s+period',
                    r'within\s+\d+\s+days',
                    r'within\s+limitation'
                ])
            },
        ],
        'required_evidence': [
            {
                'id': 'appeal_memo',
                'description': 'Copy of appeal memo or receipt acknowledgement',
                'patterns': _compile_patterns([
                    r'appeal\s+memo',
                    r'receipt\s+of\s+appeal'
                ])
            },
            {
                'id': 'order_under_challenge',
                'description': 'Certified copy of impugned order attached',
                'patterns': _compile_patterns([
                    r'certified\s+copy\s+of\s+order',
                    r'impugned\s+order\s+annexed'
                ])
            },
        ],
    },
]


class NERExtractor:
    """Named Entity Recognition extractor for legal documents"""

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize NER extractor with spaCy model

        Args:
            model_name: spaCy model to use (default: en_core_web_sm)
        """
        self.model_name = model_name
        self.nlp = None

        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model_name)
                print(f"✅ Loaded spaCy model: {model_name}")
            except OSError:
                print(
                    f"⚠️  spaCy model '{model_name}' not found. Attempting installation via uv…"
                )
                installed = _ensure_model_installed(
                    model_name,
                    getattr(spacy, "__version__", None)
                )
                if installed:
                    try:
                        self.nlp = spacy.load(model_name)
                        print(f"✅ Installed spaCy model: {model_name}")
                    except OSError:
                        print(
                            f"❌ Failed to load spaCy model '{model_name}' after installation. "
                            "Using fallback extraction."
                        )
                        self.nlp = None
                else:
                    print(
                        f"❌ Could not install spaCy model '{model_name}'. Using fallback extraction."
                    )
                    self.nlp = None

        # Legal-specific entity patterns
        self.legal_patterns = {
            'LEGAL_TERM': [
                r'\b(?:plaintiff|defendant|petitioner|respondent|appellant|appellee)\b',
                r'\b(?:arbitration|mediation|conciliation|adjudication)\b',
                r'\b(?:jurisdiction|venue|cause of action|relief|damages)\b',
                r'\b(?:breach|negligence|tort|contract|agreement|covenant)\b',
                r'\b(?:injunction|restraining order|writ|mandamus|certiorari)\b',
            ],
            'STATUTE_REF': [
                r'(?:Section|Sec\.|Article|Art\.)\s+\d+[A-Z]?(?:\(\d+\))?(?:\([a-z]\))?',
                r'(?:Chapter|Part)\s+(?:[IVXLCDM]+|\d+)',
                r'(?:Schedule|Sched\.)\s+(?:[IVXLCDM]+|\d+)',
                r'(?:Rule|Regulation)\s+\d+',
            ],
            'DATE_PATTERN': [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
                r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}',
            ],
            'MONETARY': [
                r'Rs\.?\s*\d+(?:,\d{3})*(?:\.\d{2})?',
                r'PKR\s*\d+(?:,\d{3})*(?:\.\d{2})?',
                r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:rupees|Rupees)',
            ],
            'LOCATION': [
                r'\b(?:District|Tehsil|Union Council)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
                r'\b(?:KP|KPK|Khyber Pakhtunkhwa|NWFP)\b',
            ]
        }

    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract named entities from text

        Args:
            text: Input text to analyze

        Returns:
            Dictionary of entity types and their occurrences
        """
        entities = defaultdict(list)

        if not text:
            return dict(entities)

        # Use spaCy if available
        if self.nlp is not None:
            doc = self.nlp(text)

            for ent in doc.ents:
                entities[ent.label_].append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 'high'
                })

        # Add legal-specific patterns
        for entity_type, patterns in self.legal_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities[entity_type].append({
                        'text': match.group(0),
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 'medium'
                    })

        # Deduplicate entities
        for entity_type in entities:
            seen = set()
            unique_entities = []
            for ent in entities[entity_type]:
                key = (ent['text'].lower(), ent['start'])
                if key not in seen:
                    seen.add(key)
                    unique_entities.append(ent)
            entities[entity_type] = unique_entities

        return dict(entities)

    def extract_parties(self, text: str) -> Dict[str, List[str]]:
        """
        Extract parties involved in the legal matter

        Args:
            text: Input text

        Returns:
            Dictionary with petitioner and respondent names
        """
        parties = {
            'petitioners': [],
            'respondents': [],
            'witnesses': [],
            'authorities': []
        }

        # Extract entities first
        entities = self.extract_entities(text)

        # Look for person and organization entities
        persons = entities.get('PERSON', [])
        orgs = entities.get('ORG', [])

        # Pattern matching for role identification
        text_lower = text.lower()

        for person in persons:
            person_text = person['text']
            # Check context around person mention
            start = max(0, person['start'] - 50)
            end = min(len(text), person['end'] + 50)
            context = text[start:end].lower()

            if any(role in context for role in ['petitioner', 'plaintiff', 'complainant']):
                parties['petitioners'].append(person_text)
            elif any(role in context for role in ['respondent', 'defendant', 'accused']):
                parties['respondents'].append(person_text)
            elif any(role in context for role in ['witness', 'testified']):
                parties['witnesses'].append(person_text)

        for org in orgs:
            org_text = org['text']
            start = max(0, org['start'] - 50)
            end = min(len(text), org['end'] + 50)
            context = text[start:end].lower()

            if any(auth in context for auth in ['government', 'department', 'authority', 'council']):
                parties['authorities'].append(org_text)

        # Remove duplicates
        for key in parties:
            parties[key] = list(set(parties[key]))

        return parties


class ClaimExtractor:
    """Extract legal claims and assertions from narrative and petition"""

    def __init__(self):
        """Initialize claim extractor"""
        # Claim indicator patterns
        self.claim_patterns = {
            'allegation': [
                r'(?:allegedly|accused of|charged with|blamed for)\s+(?P<content>.{20,200}?)(?:\.|;|\n)',
                r'(?:it is alleged that|the allegation is that)\s+(?P<content>.{20,200}?)(?:\.|;|\n)',
            ],
            'demand': [
                r'(?:demands?|seeks?|requests?|prays? for)\s+(?P<content>.{20,200}?)(?:\.|;|\n)',
                r'(?:claiming|entitled to)\s+(?P<content>.{20,200}?)(?:\.|;|\n)',
            ],
            'violation': [
                r'(?:violated|breached|contravened|infringed)\s+(?P<content>.{20,200}?)(?:\.|;|\n)',
                r'(?:violation of|breach of|contravention of)\s+(?P<content>.{20,200}?)(?:\.|;|\n)',
            ],
            'relief': [
                r'(?:relief|remedy|compensation|damages)\s+(?:sought|claimed|requested)\s+(?P<content>.{20,200}?)(?:\.|;|\n)',
                r'(?:seeking|requesting)\s+(?:relief|remedy|compensation)\s+(?P<content>.{20,200}?)(?:\.|;|\n)',
            ],
            'fact': [
                r'(?:on|dated?|at)\s+(?P<date>\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+(?P<content>.{20,200}?)(?:\.|;|\n)',
                r'(?:fact that|factually|in fact)\s+(?P<content>.{20,200}?)(?:\.|;|\n)',
            ]
        }

        # Temporal indicators
        self.temporal_patterns = [
            r'(?:on|dated?|at|during|from|to|until|since|before|after)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})',
        ]

    def extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract legal claims from text

        Args:
            text: Input text (narrative or petition)

        Returns:
            List of extracted claims with metadata
        """
        claims = []

        if not text:
            return claims

        for claim_type, patterns in self.claim_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    groups = match.groupdict()
                    claim_text = groups.get('content')
                    if claim_text is None:
                        if match.lastindex:
                            # Prefer the last captured group if no named content was provided
                            claim_text = match.group(match.lastindex)
                        else:
                            claim_text = match.group(0)
                    if claim_text is None:
                        continue
                    claims.append({
                        'type': claim_type,
                        'text': claim_text.strip(),
                        'position': match.start(),
                        'confidence': 0.8,
                        'date_reference': groups.get('date'),
                        'extracted_at': datetime.now().isoformat()
                    })

        return claims

    def extract_chronology(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract chronological events from text

        Args:
            text: Input text

        Returns:
            List of events with dates and descriptions
        """
        events = []

        for pattern in self.temporal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract date
                date_text = match.group(1) if match.lastindex else match.group(0)

                # Extract surrounding context (event description)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 150)
                context = text[start:end].strip()

                events.append({
                    'date': date_text,
                    'description': context,
                    'position': match.start()
                })

        # Sort by position in text (chronological order as presented)
        events.sort(key=lambda x: x['position'])

        return events

    def identify_inconsistencies(
        self,
        narrative: str,
        petition: str
    ) -> List[Dict[str, Any]]:
        """
        Identify potential inconsistencies between narrative and petition

        Args:
            narrative: User's narrative text
            petition: Opponent's petition text

        Returns:
            List of potential inconsistencies
        """
        inconsistencies = []

        # Extract claims from both documents
        narrative_claims = self.extract_claims(narrative)
        petition_claims = self.extract_claims(petition)

        # Extract chronologies
        narrative_events = self.extract_chronology(narrative)
        petition_events = self.extract_chronology(petition)

        # Compare factual claims
        narrative_facts = [c for c in narrative_claims if c['type'] == 'fact']
        petition_facts = [c for c in petition_claims if c['type'] == 'fact']

        # Check for contradictory dates
        narrative_dates = {e['date'] for e in narrative_events}
        petition_dates = {e['date'] for e in petition_events}

        if narrative_dates and petition_dates:
            # Look for date mismatches (simple heuristic)
            if len(narrative_dates.symmetric_difference(petition_dates)) > 0:
                inconsistencies.append({
                    'type': 'date_mismatch',
                    'description': 'Dates mentioned in narrative differ from petition',
                    'narrative_dates': list(narrative_dates),
                    'petition_dates': list(petition_dates),
                    'severity': 'medium'
                })

        # Check for contradictory allegations
        narrative_allegations = [c['text'] for c in narrative_claims if c['type'] == 'allegation']
        petition_allegations = [c['text'] for c in petition_claims if c['type'] == 'allegation']

        if len(narrative_allegations) < len(petition_allegations) // 2:
            inconsistencies.append({
                'type': 'missing_counter_allegations',
                'description': 'Narrative addresses fewer allegations than presented in petition',
                'petition_count': len(petition_allegations),
                'narrative_count': len(narrative_allegations),
                'severity': 'high'
            })

        return inconsistencies

    def extract_demands(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract specific demands/relief sought

        Args:
            text: Input text

        Returns:
            List of demands with classification
        """
        demands = []

        # Look for prayer/relief sections
        prayer_match = re.search(
            r'(?:PRAYER|RELIEF|WHEREFORE|DEMANDS?)[\s:]+(.+?)(?=\n\n|\Z)',
            text,
            re.IGNORECASE | re.DOTALL
        )

        if prayer_match:
            prayer_text = prayer_match.group(1)

            # Extract numbered demands
            numbered_demands = re.finditer(
                r'(?:^|\n)\s*(?:\d+\.|[a-z]\))\s*(.+?)(?=\n\s*(?:\d+\.|[a-z]\))|$)',
                prayer_text,
                re.MULTILINE | re.DOTALL
            )

            for match in numbered_demands:
                demand_text = match.group(1).strip()
                demands.append({
                    'text': demand_text,
                    'category': self._categorize_demand(demand_text),
                    'source': 'prayer_section'
                })

        # Also extract demands from main text
        claim_demands = [c for c in self.extract_claims(text) if c['type'] in ['demand', 'relief']]
        for claim in claim_demands:
            demands.append({
                'text': claim['text'],
                'category': self._categorize_demand(claim['text']),
                'source': 'main_text'
            })

        return demands

    def _categorize_demand(self, demand_text: str) -> str:
        """Categorize demand by type"""
        text_lower = demand_text.lower()

        if any(word in text_lower for word in ['compensat', 'damag', 'money', 'rs', 'amount']):
            return 'monetary'
        elif any(word in text_lower for word in ['injunction', 'restrain', 'prohibit', 'cease']):
            return 'injunctive'
        elif any(word in text_lower for word in ['declaration', 'declar']):
            return 'declaratory'
        elif any(word in text_lower for word in ['specific performance', 'compel', 'enforce']):
            return 'specific_performance'
        else:
            return 'other'


class SectionRouter:
    """Deterministic mapper enforcing subject-matter accuracy and procedures."""

    SECTION_PATTERN = re.compile(r'(?:Section|Sec\.)\s+(\d+[A-Z]?)(?:\(([^)]+)\))*', re.IGNORECASE)
    ARTICLE_PATTERN = re.compile(r'(?:Article|Art\.)\s+(\d+[A-Z]?)', re.IGNORECASE)
    SCHEDULE_PATTERN = re.compile(r'(?:Schedule|Sched\.)\s+(\d+|[IVXLCDM]+)', re.IGNORECASE)

    def __init__(self, window: int = 180):
        self.window = window
        self.known_sections = _KNOWN_SECTION_BASES
        self.rules: List[Dict[str, Any]] = []
        for rule in ACTION_ROUTER_RULES:
            compiled = {
                'id': rule['id'],
                'description': rule['description'],
                'domain': rule['domain'],
                'expected_sections': rule['expected_sections'],
                'regex': re.compile(rule['regex'], re.IGNORECASE)
                if isinstance(rule['regex'], str)
                else rule['regex'],
                'mandatory_steps': rule.get('mandatory_steps', []),
                'required_evidence': rule.get('required_evidence', []),
            }
            self.rules.append(compiled)

    def evaluate(
        self,
        narrative_text: str,
        petition_text: str,
        narrative_data: Dict[str, Any],
        petition_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate text for section routing, domain alignment, and procedures."""
        combined_text = f"{narrative_text}\n{petition_text}".strip()
        citations = self._collect_statute_refs(narrative_data, 'narrative') + self._collect_statute_refs(petition_data, 'petition')
        section_citations = [ref for ref in citations if ref['type'] == 'section']
        hallucinated_sections = sorted({ref['identifier'] for ref in section_citations if ref['identifier'] not in self.known_sections})
        total_section_citations = len(section_citations)

        action_occurrences: Dict[str, Dict[str, Any]] = {}
        for rule in self.rules:
            action_occurrences[rule['id']] = {
                'rule': rule,
                'occurrences': []
            }

        for source, text in (('narrative', narrative_text), ('petition', petition_text)):
            if not text:
                continue
            for rule in self.rules:
                for match in rule['regex'].finditer(text):
                    start, end = match.span()
                    context = text[max(0, start - self.window): min(len(text), end + self.window)]
                    action_occurrences[rule['id']]['occurrences'].append({
                        'source': source,
                        'start': start,
                        'end': end,
                        'context': context,
                        'match_text': match.group(0)
                    })

        action_results: List[Dict[str, Any]] = []
        evidence_callouts: List[Dict[str, Any]] = []
        wrong_domain_section_set: Set[str] = set()

        for rule_id, payload in list(action_occurrences.items()):
            rule = payload['rule']
            occurrences = payload['occurrences']
            if not occurrences:
                continue

            combined_context = " ".join(occ['context'] for occ in occurrences)
            citations_for_action = self._citations_for_occurrences(citations, occurrences)
            expected_set = set(rule['expected_sections'])
            present_expected = sorted({ref['identifier'] for ref in citations_for_action if ref['type'] == 'section' and ref['identifier'] in expected_set})
            present_expected_global = sorted({ref['identifier'] for ref in section_citations if ref['identifier'] in expected_set})
            missing_expected = sorted(expected_set.difference(set(present_expected_global)))

            wrong_domain_sections = sorted({
                ref['identifier']
                for ref in citations_for_action
                if ref['type'] == 'section' and self._get_section_domain(ref['identifier']) not in {rule['domain'], 'general'}
            })

            mis_cited_sections = sorted({
                ref['identifier']
                for ref in citations_for_action
                if ref['type'] == 'section' and ref['identifier'] not in expected_set
            })

            missing_steps = self._missing_items(rule['mandatory_steps'], combined_context, combined_text)
            missing_evidence = self._missing_items(rule['required_evidence'], combined_context, combined_text)

            status = 'pass'
            details: List[str] = []

            if missing_expected:
                status = 'fail'
                details.append(
                    f"Expected citation(s) {', '.join(sorted(missing_expected))} not found; route to controlling section required."
                )
            elif wrong_domain_sections:
                status = 'fail'
                details.append(
                    f"Wrong-domain citation(s) detected: {', '.join(wrong_domain_sections)}; restrict to {', '.join(rule['expected_sections'])}."
                )
            elif missing_steps:
                status = 'fail'
                details.append(
                    "Procedural safeguards missing: " + ", ".join(missing_steps)
                )
            elif missing_evidence:
                status = 'warn'
                details.append(
                    "Documentary proof required for: " + ", ".join(missing_evidence)
                )

            if missing_evidence:
                evidence_callouts.append({
                    'action': rule['description'],
                    'missing_evidence': missing_evidence
                })

            wrong_domain_section_set.update(wrong_domain_sections)

            action_results.append({
                'id': rule['id'],
                'description': rule['description'],
                'status': status,
                'expected_sections': sorted(rule['expected_sections']),
                'present_expected_sections': present_expected_global,
                'mis_cited_sections': mis_cited_sections,
                'wrong_domain_sections': wrong_domain_sections,
                'missing_steps': missing_steps,
                'missing_evidence': missing_evidence,
                'details': details,
                'contexts': [occ['context'].strip() for occ in occurrences[:3]]
            })

        total_actions = len(action_results)
        pass_count = sum(1 for action in action_results if action['status'] == 'pass')
        warn_count = sum(1 for action in action_results if action['status'] == 'warn')
        fail_count = sum(1 for action in action_results if action['status'] == 'fail')

        section_accuracy_score = 100.0 if total_actions == 0 else round((pass_count / total_actions) * 100, 1)
        wrong_domain_total = sum(len(action['wrong_domain_sections']) for action in action_results)
        wrong_domain_rate = 0.0 if total_section_citations == 0 else round((wrong_domain_total / total_section_citations) * 100, 1)
        hallucination_rate = 0.0 if total_section_citations == 0 else round((len(hallucinated_sections) / total_section_citations) * 100, 1)

        overall_status = 'pass'
        if fail_count > 0:
            overall_status = 'fail'
        elif warn_count > 0:
            overall_status = 'warn'

        if total_actions > 0 and section_accuracy_score < 100.0:
            overall_status = 'fail'
        if wrong_domain_rate > 0.0:
            overall_status = 'fail'
        if hallucination_rate > 0.0:
            overall_status = 'fail'

        metrics = {
            'detected_actions': total_actions,
            'passes': pass_count,
            'warns': warn_count,
            'fails': fail_count,
            'section_accuracy_score': section_accuracy_score,
            'wrong_domain_rate': wrong_domain_rate,
            'hallucination_rate': hallucination_rate,
            'total_section_citations': total_section_citations,
        }

        metrics_dashboard = {
            'section_accuracy_score': section_accuracy_score,
            'wrong_domain_rate': wrong_domain_rate,
            'hallucination_rate': hallucination_rate,
            'actions_evaluated': total_actions,
            'citations_checked': total_section_citations,
        }

        return {
            'actions': action_results,
            'overall_status': overall_status,
            'metrics': metrics,
            'metrics_dashboard': metrics_dashboard,
            'callouts': evidence_callouts,
            'wrong_domain_sections': sorted(wrong_domain_section_set),
            'hallucinated_sections': hallucinated_sections
        }

    def _collect_statute_refs(self, data: Dict[str, Any], source: str) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        for entry in data.get('entities', {}).get('STATUTE_REF', []):
            raw_text = entry.get('text', '')
            if not raw_text:
                continue
            parsed = self._parse_reference(raw_text)
            parsed.update({
                'raw': raw_text,
                'source': source,
                'start': entry.get('start'),
                'end': entry.get('end')
            })
            refs.append(parsed)
        return refs

    def _parse_reference(self, text: str) -> Dict[str, Any]:
        section_match = self.SECTION_PATTERN.search(text)
        if section_match:
            identifier = section_match.group(1).upper()
            return {
                'type': 'section',
                'identifier': identifier,
                'subsections': section_match.group(2)
            }
        article_match = self.ARTICLE_PATTERN.search(text)
        if article_match:
            return {
                'type': 'article',
                'identifier': article_match.group(1).upper(),
                'subsections': None
            }
        schedule_match = self.SCHEDULE_PATTERN.search(text)
        if schedule_match:
            return {
                'type': 'schedule',
                'identifier': schedule_match.group(1).upper(),
                'subsections': None
            }
        return {
            'type': 'unknown',
            'identifier': text,
            'subsections': None
        }

    def _citations_for_occurrences(
        self,
        citations: List[Dict[str, Any]],
        occurrences: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        window_citations: List[Dict[str, Any]] = []
        seen = set()
        for occ in occurrences:
            for ref in citations:
                if ref['source'] != occ['source']:
                    continue
                ref_start = ref.get('start')
                ref_end = ref.get('end')
                if ref_start is None or ref_end is None:
                    continue
                if ref_start >= occ['start'] - self.window and ref_start <= occ['end'] + self.window:
                    key = (ref['source'], ref_start, ref.get('identifier'))
                    if key not in seen:
                        seen.add(key)
                        window_citations.append(ref)
        return window_citations

    def _missing_items(
        self,
        checklist: List[Dict[str, Any]],
        context_text: str,
        full_text: str
    ) -> List[str]:
        missing: List[str] = []
        for item in checklist:
            patterns: List[re.Pattern] = item.get('patterns', [])
            if not patterns:
                continue
            if not self._patterns_present(patterns, context_text, full_text):
                missing.append(item['description'])
        return missing

    def _patterns_present(
        self,
        patterns: List[re.Pattern],
        primary_text: str,
        fallback_text: str
    ) -> bool:
        for pattern in patterns:
            if pattern.search(primary_text):
                return True
        if fallback_text is not primary_text:
            for pattern in patterns:
                if pattern.search(fallback_text):
                    return True
        return False

    def _get_section_domain(self, identifier: str) -> str:
        match = re.match(r'(\d+)', identifier)
        if not match:
            return 'general'
        base = int(match.group(1))
        for domain, section_set in SECTION_DOMAIN_RANGES.items():
            if base in section_set:
                return domain
        return 'general'

class LegalQualityAnalyzer:
    """Heuristic quality checks for legal pleadings."""

    def __init__(self):
        """Initialize keyword dictionaries and weighting scheme."""
        # Patterns to validate statutory anchors
        self.statute_anchor_pattern = re.compile(
            r'(Section|Sec\.|Article|Art\.|Rule|Regulation|Clause)\s+\d+[A-Z]?(?:\([0-9a-z]+\))*',
            re.IGNORECASE
        )
        self.act_pattern = re.compile(
            r'([A-Z][A-Za-z\s]+ Act(?:,)? \d{4})',
            re.IGNORECASE
        )
        self.article_pattern = re.compile(
            r'(Article|Art\.)\s+\d+[A-Z]?(?:\([0-9a-z]+\))*',
            re.IGNORECASE
        )
        self.named_date_pattern = re.compile(
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
            re.IGNORECASE
        )
        self.numeric_date_pattern = re.compile(
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        )

        # Domain keywords
        self.fairness_keywords = [
            'natural justice', 'due process', 'fair hearing', 'audi alteram partem',
            'fair trial', 'procedural fairness', 'right to be heard'
        ]
        self.procedural_keywords = [
            'appeal', 'appellate', 'review', 'revision', 'representation',
            'grievance redressal', 'complaint forum', 'tribunal', 'ombudsman',
            'internal remedy'
        ]
        self.evidence_keywords = [
            'notice', 'order', 'letter', 'correspondence', 'annexure', 'annexed',
            'exhibit', 'attachment', 'record', 'document', 'minutes', 'transcript'
        ]
        self.topic_keywords: Dict[str, List[str]] = {
            'taxation': ['tax', 'levy', 'assessment', 'duty', 'excise'],
            'suspension': ['suspend', 'suspension', 'remove', 'removal', 'dismiss', 'termination'],
            'elections': ['election', 'ballot', 'poll', 'nomination', 'ward', 'campaign'],
            'finance': ['budget', 'grant', 'fund', 'audit', 'accounts', 'finance'],
            'procurement': ['tender', 'procurement', 'bid', 'contract'],
            'discipline': ['misconduct', 'disciplinary', 'show-cause', 'enquiry', 'inquiry'],
            'land': ['acquisition', 'land', 'property', 'allotment', 'encroachment']
        }

        self.section_router = SectionRouter()

        # Weighted impact (percentage weights sum to 100)
        self.metric_weights = {
            'section_router': 20,
            'statutory_accuracy': 20,
            'grounding_principles': 15,
            'pruning_irrelevant_law': 10,
            'procedural_remedies': 10,
            'evidence_trail': 15,
            'constitutional_framing': 5,
            'narrative_discipline': 5,
        }
        self.status_score = {'pass': 1.0, 'warn': 0.5, 'fail': 0.0}

    def evaluate(
        self,
        narrative_text: str,
        petition_text: str,
        narrative_data: Dict[str, Any],
        petition_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run all quality checks and return structured results."""
        combined_text = f"{narrative_text}\n\n{petition_text}".strip()
        router_result = self.section_router.evaluate(
            narrative_text,
            petition_text,
            narrative_data,
            petition_data
        )
        statute_refs = self._collect_statute_refs(narrative_data, petition_data)

        fairness_contexts = self._find_keyword_contexts(combined_text, self.fairness_keywords)
        article_matches = list(self.article_pattern.finditer(combined_text))

        metrics = {
            'section_router': self._build_router_metric(router_result),
            'statutory_accuracy': self._check_statutory_accuracy(statute_refs),
            'grounding_principles': self._check_principles_grounding(fairness_contexts, combined_text),
            'pruning_irrelevant_law': self._check_pruning_irrelevant(
                statute_refs,
                combined_text,
                router_result
            ),
            'procedural_remedies': self._check_procedural_remedies(combined_text),
            'evidence_trail': self._check_evidence_trail(narrative_data, petition_data, narrative_text),
            'constitutional_framing': self._check_constitutional_framing(
                article_matches,
                combined_text,
                statute_refs
            ),
            'narrative_discipline': self._check_narrative_discipline(narrative_text, petition_text)
        }

        weighted_score = self._calculate_weighted_score(metrics)
        metrics_summary = {
            'weighted_score_percent': weighted_score,
            'flagged_metrics': [name for name, metric in metrics.items() if metric['status'] != 'pass'],
            'passing_metrics': [name for name, metric in metrics.items() if metric['status'] == 'pass'],
            'metrics_dashboard': router_result['metrics_dashboard'],
            'router_overall_status': router_result['overall_status'],
            'evidence_callouts': router_result['callouts']
        }

        return {
            'metrics': metrics,
            'summary': metrics_summary,
            'router': router_result
        }

    def _build_router_metric(self, router_result: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize section routing results as a metric entry."""
        status = router_result.get('overall_status', 'fail')
        actions = router_result.get('actions', [])

        failing = [a for a in actions if a['status'] == 'fail']
        warnings = [a for a in actions if a['status'] == 'warn']

        if not actions:
            summary = 'No high-risk statutory actions detected.'
        elif failing:
            summary = 'Routing failed for: ' + ", ".join(a['description'] for a in failing)
        elif warnings:
            summary = 'Routing warnings: ' + ", ".join(a['description'] for a in warnings)
        else:
            summary = 'All detected actions cite controlling provisions with mandatory safeguards.'

        details = {
            'metrics_dashboard': router_result.get('metrics_dashboard', {}),
            'callouts': router_result.get('callouts', []),
            'actions': [
                {
                    'description': action['description'],
                    'status': action['status'],
                    'details': action['details'],
                    'expected_sections': action['expected_sections'],
                    'present_sections': action['present_expected_sections'],
                    'mis_cited_sections': action['mis_cited_sections'],
                    'missing_steps': action['missing_steps'],
                    'missing_evidence': action['missing_evidence']
                }
                for action in actions
            ]
        }

        return {
            'status': status,
            'summary': summary,
            'details': details
        }

    def _collect_statute_refs(
        self,
        narrative_data: Dict[str, Any],
        petition_data: Dict[str, Any]
    ) -> List[str]:
        """Gather statute references from processed data."""
        refs: List[str] = []
        for data in (narrative_data, petition_data):
            for entry in data.get('entities', {}).get('STATUTE_REF', []):
                text = entry.get('text', '').strip()
                if text:
                    refs.append(text)
        return refs

    def _check_statutory_accuracy(self, statute_refs: List[str]) -> Dict[str, Any]:
        """Validate statutory references for format and potential padding."""
        if not statute_refs:
            return {
                'status': 'fail',
                'summary': 'No statutory citations detected. Verify governing provisions.',
                'details': []
            }

        invalid_refs = []
        padding_refs = []
        normalized_refs = set()
        for ref in statute_refs:
            normalized = re.sub(r'\s+', ' ', ref.strip().lower())
            normalized_refs.add(normalized)
            if not self.statute_anchor_pattern.search(ref):
                invalid_refs.append(ref)
            if 'schedule' in ref.lower() or 'part ' in ref.lower():
                padding_refs.append(ref)

        if invalid_refs:
            return {
                'status': 'fail',
                'summary': 'Some citations look malformed. Confirm section numbering.',
                'details': invalid_refs
            }

        if padding_refs:
            return {
                'status': 'warn',
                'summary': 'Citations include schedules/parts. Ensure they directly govern the dispute.',
                'details': padding_refs
            }

        return {
            'status': 'pass',
            'summary': f'{len(normalized_refs)} statutory citation(s) with clean formatting.',
            'details': sorted(normalized_refs)
        }

    def _check_principles_grounding(
        self,
        fairness_contexts: List[Dict[str, Any]],
        combined_text: str
    ) -> Dict[str, Any]:
        """Ensure fairness arguments are anchored to statute."""
        if not fairness_contexts:
            return {
                'status': 'pass',
                'summary': 'No fairness rhetoric detected without statutes.',
                'details': []
            }

        anchored = []
        floating = []
        for ctx in fairness_contexts:
            if self.statute_anchor_pattern.search(ctx['context']):
                anchored.append(ctx['keyword'])
            else:
                backtrack = combined_text[max(0, ctx['start'] - 160):ctx['start']]
                if self.statute_anchor_pattern.search(backtrack):
                    anchored.append(ctx['keyword'])
                else:
                    floating.append(ctx['keyword'])

        if floating and not anchored:
            return {
                'status': 'fail',
                'summary': 'Fairness principles invoked without tying them to statutory duties.',
                'details': floating
            }

        if floating:
            return {
                'status': 'warn',
                'summary': 'Some fairness arguments need explicit statutory anchors.',
                'details': floating
            }

        return {
            'status': 'pass',
            'summary': 'Fairness principles located near statutory references.',
            'details': anchored
        }

    def _check_pruning_irrelevant(
        self,
        statute_refs: List[str],
        combined_text: str,
        router_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect over-broad statutory padding or topic drift."""
        wrong_domain_sections = router_result.get('wrong_domain_sections', [])
        hallucinated_sections = router_result.get('hallucinated_sections', [])
        schedule_refs = [ref for ref in statute_refs if 'schedule' in ref.lower()]
        acts = {match.group(0).strip() for match in self.act_pattern.finditer(combined_text)}
        topics = self._detect_topics(combined_text)

        status = 'pass'
        summary = 'Citations stay focused on core statute.'
        details: List[str] = []

        topic_count = len(topics)
        act_count = len(acts)

        if wrong_domain_sections:
            status = 'fail'
            summary = 'Wrong-domain citations detected. Replace with controlling provisions.'
            details.append(f"Wrong-domain sections: {', '.join(wrong_domain_sections)}")
        elif hallucinated_sections:
            status = 'fail'
            summary = 'Unrecognized section numbers cited. Verify statutory basis.'
            details.append(f"Unrecognized sections: {', '.join(hallucinated_sections)}")
        elif topic_count > 4 or act_count > 3:
            status = 'fail'
            summary = 'Narrative spans multiple unrelated legal themes. Consider pruning.'
        elif topic_count > 3 or act_count > 2 or schedule_refs:
            status = 'warn'
            summary = 'Some citations or themes look tangential. Trim to the governing provisions.'

        if acts:
            details.append(f"Acts referenced: {', '.join(sorted(acts))}")
        if topics:
            details.append(f"Themes detected: {', '.join(sorted(topics))}")
        if schedule_refs:
            details.append(f"Schedule references: {', '.join(schedule_refs)}")

        return {
            'status': status,
            'summary': summary,
            'details': details
        }

    def _check_procedural_remedies(self, combined_text: str) -> Dict[str, Any]:
        """Check for statutory appeal/review remedies."""
        contexts = self._find_keyword_contexts(combined_text, self.procedural_keywords, window=80)
        if not contexts:
            return {
                'status': 'fail',
                'summary': 'No appeal or review mechanism acknowledged. Address alternate remedies.',
                'details': []
            }

        if len(contexts) == 1:
            snippet = contexts[0]['context'].strip()
            return {
                'status': 'warn',
                'summary': 'Only a single mention of procedural remedies. Clarify why writ jurisdiction applies.',
                'details': [snippet]
            }

        snippets = [ctx['context'].strip() for ctx in contexts[:3]]
        return {
            'status': 'pass',
            'summary': 'Procedural remedies acknowledged with supporting detail.',
            'details': snippets
        }

    def _check_evidence_trail(
        self,
        narrative_data: Dict[str, Any],
        petition_data: Dict[str, Any],
        narrative_text: str
    ) -> Dict[str, Any]:
        """Evaluate chronology depth and documentary references."""
        narrative_events = len(narrative_data.get('chronology', []))
        petition_events = len(petition_data.get('chronology', []))
        evidence_hits = self._find_keyword_contexts(narrative_text, self.evidence_keywords, window=0)
        date_hits = len(self.named_date_pattern.findall(narrative_text)) + len(
            self.numeric_date_pattern.findall(narrative_text)
        )

        details = [
            f"Narrative events: {narrative_events}",
            f"Petition events: {petition_events}",
            f"Evidence markers: {len(evidence_hits)}",
            f"Date mentions: {date_hits}"
        ]

        if narrative_events >= 3 or (narrative_events >= 2 and len(evidence_hits) >= 2) or date_hits >= 4:
            return {
                'status': 'pass',
                'summary': 'Chronology and exhibits look documented.',
                'details': details
            }

        if narrative_events >= 1 or date_hits >= 1:
            return {
                'status': 'warn',
                'summary': 'Chronology is thin. Add dates and annexures for each allegation.',
                'details': details
            }

        return {
            'status': 'fail',
            'summary': 'No dated evidence trail detected. Courts expect notices, orders, and annexures.',
            'details': details
        }

    def _check_constitutional_framing(
        self,
        article_matches: List[re.Match],
        combined_text: str,
        statute_refs: List[str]
    ) -> Dict[str, Any]:
        """Assess whether constitutional claims are tied to statutes."""
        if not article_matches:
            if statute_refs:
                return {
                    'status': 'warn',
                    'summary': 'No constitutional hook cited. Consider derivative reliance on Article 10A or Article 199.',
                    'details': []
                }
            return {
                'status': 'warn',
                'summary': 'Neither statutes nor constitutional safeguards referenced. Establish a legal basis.',
                'details': []
            }

        unanchored = []
        anchored = []
        for match in article_matches:
            start, end = match.span()
            context = combined_text[max(0, start - 160):min(len(combined_text), end + 160)]
            if self.statute_anchor_pattern.search(context):
                anchored.append(match.group(0))
            else:
                unanchored.append(match.group(0))

        if unanchored and not anchored:
            return {
                'status': 'fail',
                'summary': 'Constitutional provisions cited without showing the statutory breach that triggers them.',
                'details': unanchored
            }

        if unanchored:
            return {
                'status': 'warn',
                'summary': 'Some constitutional claims lack explicit statutory linkage.',
                'details': unanchored
            }

        return {
            'status': 'pass',
            'summary': 'Constitutional provisions are tied to statutory duties.',
            'details': anchored
        }

    def _check_narrative_discipline(self, narrative_text: str, petition_text: str) -> Dict[str, Any]:
        """Ensure the narrative stays focused on core dispute."""
        topics = self._detect_topics(narrative_text + '\n' + petition_text)

        if len(topics) <= 3:
            return {
                'status': 'pass',
                'summary': 'Narrative focused on core dispute without issue sprawl.',
                'details': sorted(topics)
            }

        if len(topics) == 4:
            return {
                'status': 'warn',
                'summary': 'Multiple issue clusters detected. Keep petition laser-focused.',
                'details': sorted(topics)
            }

        return {
            'status': 'fail',
            'summary': 'Narrative mixes too many legal issues. Split or prune arguments.',
            'details': sorted(topics)
        }

    def _find_keyword_contexts(
        self,
        text: str,
        keywords: List[str],
        window: int = 120
    ) -> List[Dict[str, Any]]:
        """Locate keyword occurrences with surrounding context."""
        contexts: List[Dict[str, Any]] = []
        lower_text = text.lower()
        for keyword in keywords:
            lowered_keyword = keyword.lower()
            for match in re.finditer(re.escape(lowered_keyword), lower_text):
                start = max(0, match.start() - window)
                end = min(len(text), match.end() + window)
                contexts.append({
                    'keyword': keyword,
                    'context': text[start:end],
                    'start': match.start(),
                    'end': match.end()
                })
        return contexts

    def _detect_topics(self, text: str) -> List[str]:
        """Identify thematic clusters in the narrative."""
        topics_hit: List[str] = []
        lower_text = text.lower()
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in lower_text for keyword in keywords):
                topics_hit.append(topic)
        return topics_hit

    def _calculate_weighted_score(self, metrics: Dict[str, Dict[str, Any]]) -> float:
        """Compute weighted score based on pass/warn/fail status."""
        total_weight = sum(self.metric_weights.values())
        if total_weight == 0:
            return 0.0

        score = 0.0
        for name, weight in self.metric_weights.items():
            status = metrics.get(name, {}).get('status', 'fail')
            score += weight * self.status_score.get(status, 0.0)

        return round((score / total_weight) * 100, 1)


class DocumentProcessor:
    """Main processor that combines NER and Claim extraction"""

    def __init__(self):
        """Initialize document processor"""
        self.ner_extractor = NERExtractor()
        self.claim_extractor = ClaimExtractor()
        self.quality_analyzer = LegalQualityAnalyzer()

    def process_narrative(self, narrative: str) -> Dict[str, Any]:
        """
        Process user's narrative

        Args:
            narrative: User's narrative text

        Returns:
            Structured data with entities, claims, and chronology
        """
        return {
            'entities': self.ner_extractor.extract_entities(narrative),
            'parties': self.ner_extractor.extract_parties(narrative),
            'claims': self.claim_extractor.extract_claims(narrative),
            'chronology': self.claim_extractor.extract_chronology(narrative),
            'demands': self.claim_extractor.extract_demands(narrative),
            'processed_at': datetime.now().isoformat()
        }

    def process_petition(self, petition: str) -> Dict[str, Any]:
        """
        Process opponent's petition

        Args:
            petition: Opponent's petition text

        Returns:
            Structured data with entities, claims, and demands
        """
        return {
            'entities': self.ner_extractor.extract_entities(petition),
            'parties': self.ner_extractor.extract_parties(petition),
            'claims': self.claim_extractor.extract_claims(petition),
            'chronology': self.claim_extractor.extract_chronology(petition),
            'demands': self.claim_extractor.extract_demands(petition),
            'processed_at': datetime.now().isoformat()
        }

    def process_case(
        self,
        narrative: str,
        petition: str
    ) -> Dict[str, Any]:
        """
        Process complete case (narrative + petition)

        Args:
            narrative: User's narrative
            petition: Opponent's petition

        Returns:
            Combined analysis with inconsistencies highlighted
        """
        narrative_data = self.process_narrative(narrative)
        petition_data = self.process_petition(petition)
        inconsistencies = self.claim_extractor.identify_inconsistencies(narrative, petition)
        quality_checks = self.quality_analyzer.evaluate(
            narrative,
            petition,
            narrative_data,
            petition_data
        )

        metrics_dashboard = quality_checks['summary'].get('metrics_dashboard', {})

        return {
            'narrative': narrative_data,
            'petition': petition_data,
            'inconsistencies': inconsistencies,
            'analysis': {
                'narrative_claim_count': len(narrative_data['claims']),
                'petition_claim_count': len(petition_data['claims']),
                'inconsistency_count': len(inconsistencies),
                'critical_inconsistencies': [i for i in inconsistencies if i.get('severity') == 'high'],
                'quality_flag_count': len(quality_checks['summary']['flagged_metrics']),
                'section_accuracy_score': metrics_dashboard.get('section_accuracy_score'),
                'wrong_domain_rate': metrics_dashboard.get('wrong_domain_rate'),
                'hallucination_rate': metrics_dashboard.get('hallucination_rate')
            },
            'quality_checks': quality_checks,
            'processed_at': datetime.now().isoformat()
        }

    def export_json(self, data: Dict[str, Any], filepath: str) -> None:
        """Export processed data to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_json(self, filepath: str) -> Dict[str, Any]:
        """Load processed data from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


# Convenience functions for quick access
def extract_entities(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """Quick entity extraction"""
    extractor = NERExtractor()
    return extractor.extract_entities(text)


def extract_claims(text: str) -> List[Dict[str, Any]]:
    """Quick claim extraction"""
    extractor = ClaimExtractor()
    return extractor.extract_claims(text)


def process_case(narrative: str, petition: str) -> Dict[str, Any]:
    """Quick case processing"""
    processor = DocumentProcessor()
    return processor.process_case(narrative, petition)
