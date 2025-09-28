from typing import List, Dict, Any, Tuple, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import os

try:
    # CrossEncoder provides powerful pairwise re-ranking
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:
    CrossEncoder = None  # Fallback if unavailable in environment

# Lightweight replacements for NLTK + scikit-learn components to avoid
# binary incompatibilities in constrained environments (e.g., Kaggle images
# with mismatched numpy/compiled wheels).

# --- Simple tokenization, stopwords, and stemming ---
_BASIC_STOPWORDS = {
    'a','an','the','and','or','if','in','on','at','of','for','with','without','to','from','by','as',
    'is','it','this','that','these','those','be','been','being','are','was','were','am','do','does','did','doing',
    'have','has','had','having','not','no','nor','but','because','so','very','can','could','should','would','may','might','will',
    'just','also','than','then','once','such','own','same','other','more','most','some','any','each','few','many','much',
    'here','there','when','where','why','how','all','both','between','into','through','during','before','after','above','below',
    'up','down','out','over','under','again','further','until','while','about','against','among','within','across','per',
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves',
    'he','him','his','himself','she','her','hers','herself','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','whose','whenever','wherever','whatever','whichever',
    'been','being','didn','doesn','don','hadn','hasn','haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn','weren','won','wouldn'
}

def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text)

def simple_stem(token: str) -> str:
    # Very light stemming: common English suffix stripping
    if len(token) <= 3:
        return token
    for suf in ("ization","ations","ation","izers","izer","ities","ility","iveness","fulness",
                "ously","ness","ment","ments","ingly","ingly","edly","edly","ous","ive","ful","able",
                "less","ing","ed","ies","ers","er","s"):
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)]
    return token

# --- Minimal TF-IDF implementation with cosine similarity ---
class SimpleTfidfVectorizer:
    def __init__(self, min_df: int = 1, max_df: float = 1.0, max_features: Optional[int] = None):
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.vocab_: Dict[str, int] = {}
        self.idf_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = []

    def fit(self, docs: List[str]):
        N = len(docs)
        if N == 0:
            self.vocab_ = {}
            self.idf_ = np.zeros((0,), dtype=np.float32)
            self.feature_names_ = []
            return self
        # Document frequency
        df: Dict[str, int] = {}
        for doc in docs:
            terms = set(doc.split())
            for t in terms:
                df[t] = df.get(t, 0) + 1

        # Apply df thresholds
        max_df_abs = int(self.max_df * N) if 0 < self.max_df <= 1.0 else int(self.max_df)
        if max_df_abs == 0:
            max_df_abs = N
        terms = [t for t, c in df.items() if c >= self.min_df and c <= max_df_abs]

        # Top features if requested (by df descending then alpha)
        terms.sort(key=lambda t: (-df[t], t))
        if self.max_features is not None:
            terms = terms[: self.max_features]

        self.feature_names_ = terms
        self.vocab_ = {t: i for i, t in enumerate(self.feature_names_)}

        # IDF (smooth like sklearn): log((1 + N) / (1 + df)) + 1
        idf = np.ones((len(self.vocab_),), dtype=np.float32)
        for t, i in self.vocab_.items():
            idf[i] = np.log((1.0 + N) / (1.0 + df[t])) + 1.0
        self.idf_ = idf
        return self

    def transform(self, docs: List[str]) -> np.ndarray:
        V = len(self.vocab_)
        if V == 0:
            return np.zeros((len(docs), 0), dtype=np.float32)
        matrix = np.zeros((len(docs), V), dtype=np.float32)
        for row, doc in enumerate(docs):
            counts: Dict[str, int] = {}
            for term in doc.split():
                if term in self.vocab_:
                    counts[term] = counts.get(term, 0) + 1
            if not counts:
                continue
            max_tf = max(counts.values())
            for term, cnt in counts.items():
                col = self.vocab_[term]
                tf = cnt / max_tf
                matrix[row, col] = tf * self.idf_[col]
        # L2 normalize rows
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix /= norms
        return matrix

    def fit_transform(self, docs: List[str]) -> np.ndarray:
        return self.fit(docs).transform(docs)

def cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Supports shapes: (1, d) x (n, d) or (m, d) x (n, d)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if B.ndim == 1:
        B = B.reshape(1, -1)
    # Normalize
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T

class HybridSearchEngine:
    def __init__(
        self,
        embedding_model_name: str = 'all-MiniLM-L6-v2',
        embedding_model: Optional[SentenceTransformer] = None
    ):
        """Initialize hybrid search engine with semantic and keyword search"""
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model_name = embedding_model_name
        self.bm25 = None
        # Use simple internal TF-IDF to avoid scikit-learn
        self.tfidf_vectorizer = SimpleTfidfVectorizer(
            max_features=5000,
            max_df=0.85,
            min_df=2
        )
        self.tfidf_matrix = None
        self.stem = simple_stem
        self.stop_words = set(_BASIC_STOPWORDS)
        
        # Legal-specific stop words and terms
        self.legal_stop_words = {
            'shall', 'may', 'must', 'should', 'will', 'can', 'could',
            'would', 'might', 'said', 'aforesaid', 'thereof', 'wherein',
            'whereas', 'hereby', 'herein', 'hereafter', 'heretofore'
        }
        
        self.documents = []
        self.document_embeddings = None
        self.processed_docs = []

        # Optional cross-encoder re-ranker (legal-friendly if available)
        self.reranker = None
        self._init_reranker()

    def _init_reranker(self):
        """Initialise cross-encoder re-ranker with graceful fallback.

        Picks a model tuned for QA-style relevance. Allows override via
        RAGBOT_RERANKER_MODEL. If loading fails or dependency missing,
        continues without re-ranking.
        """
        if CrossEncoder is None:
            print("ℹ️ Cross-encoder not available; skipping re-ranking.")
            return
        preferred_models = [
            # Prefer a QA/click-tuned cross-encoder; users may override via env
            os.getenv("RAGBOT_RERANKER_MODEL", ""),
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "cross-encoder/stsb-roberta-base",
        ]
        tried = []
        for name in preferred_models:
            name = (name or "").strip()
            if not name:
                continue
            try:
                print(f"Loading cross-encoder reranker '{name}' ...")
                self.reranker = CrossEncoder(name)
                print(f"✅ Cross-encoder loaded: {name}")
                return
            except Exception as exc:
                tried.append((name, str(exc)))
                continue
        if tried:
            print("⚠️ Failed to load cross-encoder models:")
            for name, err in tried:
                print(f"   - {name}: {err}")
        print("ℹ️ Proceeding without cross-encoder re-ranking.")

    # -------- Query focus + precision helpers --------
    def _extract_focus_terms(self, query: str) -> List[str]:
        """Extract domain-relevant focus terms from the query.

        Returns a small, de-duplicated list of keywords (lowercased) including
        synonyms for detected intent clusters (e.g., disputes/conflicts).
        """
        q = (query or "").lower()
        terms: List[str] = []

        # Core content words: strip stop-words and short tokens
        base_tokens = [t for t in simple_tokenize(q) if len(t) > 2 and t not in self.stop_words]
        terms.extend(base_tokens)

        # Intent clusters with synonyms
        clusters = {
            # Dispute/conflict resolution
            'dispute': [
                'dispute', 'disputes', 'conflict', 'conflicts', 'resolution', 'resolve', 'settlement',
                'redress', 'grievance', 'complaint', 'appeal', 'review', 'mediation', 'arbitration',
                'conciliation', 'adjudication', 'ombudsman'
            ],
            # Elections/constituencies/delimitation
            'election': [
                'election', 'elections', 'electoral', 'poll', 'polling', 'vote', 'ballot', 'constituency',
                'delimitation', 'ward'
            ],
            # Finance/audit
            'finance': [
                'finance', 'financial', 'fund', 'funds', 'budget', 'account', 'accounts', 'accounting',
                'audit', 'auditor', 'audited', 'treasury', 'receipt', 'expenditure', 'tax', 'fee', 'fees'
            ],
            # Enforcement/penalties
            'enforcement': [
                'enforcement', 'penalty', 'penalties', 'offence', 'offense', 'fine', 'fines', 'prosecution',
                'sanction', 'violation'
            ],
        }

        for key, vocab in clusters.items():
            if any(t in q for t in vocab):
                terms.extend(vocab)

        # Deduplicate while preserving order
        seen = set()
        uniq: List[str] = []
        for t in terms:
            if t not in seen:
                uniq.append(t)
                seen.add(t)
        return uniq

    def _keyword_overlap_score(self, focus_terms: List[str], text: str, metadata: Dict[str, Any]) -> float:
        """Compute a lightweight keyword overlap score [0..1].

        Measures presence of focus terms in text/metadata. Penalises if none
        are found when focus_terms is non-empty.
        """
        if not focus_terms:
            return 0.0
        hay = f"{(text or '').lower()} \n {(str(metadata) or '').lower()}"
        hits = 0
        budget = 0
        # Limit budget to first 14 distinct focus terms to keep bounded
        for t in focus_terms[:14]:
            budget += 1
            if t in hay:
                hits += 1
        if budget == 0:
            return 0.0
        return hits / float(budget)

    def _pinpoint_spans(self, text: str, focus_terms: List[str], window: int = 1) -> Dict[str, Any]:
        """Extract minimal relevant spans and best-effort subsection labels.

        Returns a dict with:
          - sliced_text: reduced text containing only lines with focus terms +/- window
          - labels: list of subsection labels inferred like (1), (a), (i)
        """
        if not text:
            return {"sliced_text": "", "labels": []}
        lines = [ln for ln in text.split('\n')]
        focus = [t for t in focus_terms if len(t) > 2]
        hit_indices = []
        for i, ln in enumerate(lines):
            low = ln.lower()
            if any(t in low for t in focus):
                hit_indices.append(i)
        if not hit_indices:
            return {"sliced_text": "", "labels": []}
        keep = set()
        labels: List[str] = []
        # Patterns for clause labels at line-start
        p_multi = re.compile(r'^\s*(\(\d+\))?\s*(\([a-z]\))?\s*(\([ivxlcdm]+\))?', re.IGNORECASE)
        for idx in hit_indices:
            for j in range(max(0, idx - window), min(len(lines), idx + window + 1)):
                keep.add(j)
            # Try to capture a label at or above the hit line
            target = lines[idx]
            m = p_multi.match(target)
            if not any(m.groups()) if m else True:
                # look one line above for a parent label chain
                if idx > 0:
                    prev = lines[idx - 1]
                    m = p_multi.match(prev)
            if m:
                for g in m.groups():
                    if g:
                        labels.append(g.strip('()'))
        sliced_text = '\n'.join([lines[i] for i in sorted(keep)])
        # De-duplicate labels preserving order
        seen = set()
        plabels: List[str] = []
        for l in labels:
            if l not in seen:
                plabels.append(l)
                seen.add(l)
        return {"sliced_text": sliced_text, "labels": plabels}

    def apply_precision_filters(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5,
        max_results: int = 4,
        min_score: float = 0.25,
        min_overlap: float = 0.12,
    ) -> List[Dict[str, Any]]:
        """Filter and slim results to enforce surgical grounding.

        - Rank by final score, then keyword overlap.
        - Require combined focus overlap when a clear intent cluster exists.
        - De-duplicate by section/schedule to avoid citation stuffing.
        - Attach pinpoint span slices and labels for precise citations.
        """
        if not results:
            return []

        focus_terms = self._extract_focus_terms(query)
        # Determine if we have an intentful focus (e.g., disputes)
        has_focus = any(k in focus_terms for k in (
            'dispute', 'conflict', 'resolution', 'arbitration', 'mediation',
            'appeal', 'complaint', 'grievance'
        ))

        enriched: List[Dict[str, Any]] = []
        for r in results:
            md = r.get('metadata', {}) or {}
            text = r.get('text', '') or ''
            overlap = self._keyword_overlap_score(focus_terms, text, md)
            # Attach breakdown bits so downstream can surface them
            br = dict(r.get('score_breakdown', {}))
            br['focus'] = float(overlap)
            r['score_breakdown'] = br

            # Compute pinpoint spans to reduce context size
            pin = self._pinpoint_spans(text, focus_terms)
            if pin.get('sliced_text'):
                r['sliced_text'] = pin['sliced_text']
            if pin.get('labels'):
                r['pinpoint_labels'] = pin['labels']

            enriched.append(r)

        # Soft gate first by score
        gated = [r for r in enriched if float(r.get('score', 0.0)) >= min_score]
        if has_focus:
            # Apply keyword overlap threshold when focus intent detected
            gated = [r for r in gated if r.get('score_breakdown', {}).get('focus', 0.0) >= min_overlap]

        # De-duplicate by statutory unit (prefer section number, else schedule ref, else text hash)
        deduped: List[Dict[str, Any]] = []
        seen_keys = set()
        for r in sorted(gated, key=lambda x: (float(x.get('score', 0.0)), float(x.get('score_breakdown', {}).get('focus', 0.0))), reverse=True):
            md = r.get('metadata', {}) or {}
            key = None
            if md.get('section_number'):
                key = f"sec:{md.get('section_number')}"
            elif md.get('schedule_ref'):
                key = f"sch:{md.get('schedule_ref')}"
            else:
                key = f"hash:{hash(r.get('text',''))}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(r)
            if len(deduped) >= max_results:
                break

        # Fallback if all were filtered out: keep top-1 to avoid empty answers
        if not deduped and results:
            best = max(results, key=lambda x: float(x.get('score', 0.0)))
            deduped = [best]

        return deduped
        
    def preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing for legal documents"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep legal references
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        
        # Normalize legal references
        text = re.sub(r'section\s+(\d+)', r'section_\1', text)
        text = re.sub(r'article\s+(\d+)', r'article_\1', text)
        text = re.sub(r'chapter\s+(\d+)', r'chapter_\1', text)
        text = re.sub(r'part\s+(\d+)', r'part_\1', text)
        
        # Tokenize
        tokens = simple_tokenize(text)
        
        # Remove stop words and apply stemming
        processed_tokens = []
        for token in tokens:
            if (token not in self.stop_words and 
                token not in self.legal_stop_words and
                len(token) > 2):
                stemmed = self.stem(token)
                processed_tokens.append(stemmed)
        
        return ' '.join(processed_tokens)
    
    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract legal entities and references"""
        entities = {
            'sections': [],
            'articles': [],
            'chapters': [],
            'parts': [],
            'clauses': [],
            'subclauses': [],
            'schedules': [],
            'legal_terms': [],
            'numbers': []
        }
        
        # Extract section references
        sections = re.findall(r'section\s+(\d+)', text.lower())
        entities['sections'] = list(set(sections))
        
        # Extract article references
        articles = re.findall(r'article\s+(\d+)', text.lower())
        entities['articles'] = list(set(articles))
        
        # Extract chapter references
        chapters = re.findall(r'chapter\s+(\d+)', text.lower())
        entities['chapters'] = list(set(chapters))
        
        # Extract part references
        parts = re.findall(r'part\s+(\d+)', text.lower())
        entities['parts'] = list(set(parts))

        # Extract schedule references
        schedules = re.findall(r'schedule\s+([ivxlcdm]+|\d+)', text.lower())
        entities['schedules'] = list(set(schedules))

        clauses = re.findall(r'clause\s+(\d+|[a-z])', text.lower())
        entities['clauses'] = list(set(clauses))

        subclauses = re.findall(r'sub-?clause\s+(\d+|[a-z]|[ivxlcdm]+)', text.lower())
        entities['subclauses'] = list(set(subclauses))

        # Extract legal terms
        legal_patterns = [
            r'\b(?:council|committee|government|district|union|local)\b',
            r'\b(?:power|function|duty|responsibility|authority)\b',
            r'\b(?:election|electoral|member|chairman|mayor|administrator|constituency|delimitation)\b',
            r'\b(?:budget|fund|tax|fee|development|planning|accounts|audit|auditor)\b',
            r'\b(?:notification|rule|regulation|ordinance|act)\b',
            r'\b(?:dispute|conflict|resolution|complaint|grievance|appeal|mediation|arbitration|conciliation)\b'
        ]
        
        for pattern in legal_patterns:
            matches = re.findall(pattern, text.lower())
            entities['legal_terms'].extend(matches)
        
        entities['legal_terms'] = list(set(entities['legal_terms']))
        
        return entities
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for hybrid search"""
        print("🔍 Indexing documents for hybrid search...")
        
        self.documents = documents
        
        # Extract text content for indexing
        doc_texts = []
        processed_texts = []
        
        for doc in documents:
            text = doc.get('text', '')
            doc_texts.append(text)
            processed_text = self.preprocess_text(text)
            processed_texts.append(processed_text)
        
        self.processed_docs = processed_texts
        
        # Create BM25 index
        tokenized_docs = [doc.split() for doc in processed_texts]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Create TF-IDF matrix
        if processed_texts:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts)
        
        # Create semantic embeddings
        print("🧠 Generating semantic embeddings...")
        self.document_embeddings = self.embedding_model.encode(doc_texts)
        
        print(f"✅ Indexed {len(documents)} documents")

    # -------- Domain-aware ranking helpers --------
    def _is_definition_query(self, query: str) -> bool:
        q = query.lower().strip()
        indicators = [
            "define ",
            "definition of ",
            "what is the definition",
            "meaning of ",
            "interpretation of ",
        ]
        return any(tok in q for tok in indicators)

    def _penalty_for_generic(self, query: str, metadata: Dict[str, Any], text: str) -> float:
        """Return a penalty in [0, 0.25] for generic sections unless explicitly asked."""
        if self._is_definition_query(query):
            return 0.0

        title_bits = []
        for k in ("title", "section", "chapter", "part"):
            v = metadata.get(k)
            if isinstance(v, str):
                title_bits.append(v)
        title = " ".join(title_bits).lower()
        head = (text[:120] if text else "").lower()

        generic_keys = [
            "definitions", "interpretation", "preliminary", "general provisions",
            "short title", "extent", "commencement", "application", "framework",
        ]
        hay = f"{title} {head}"
        hits = sum(1 for k in generic_keys if k in hay)
        if hits == 0:
            return 0.0
        # Scale by intensity of generic signals, cap at 0.25
        return min(0.08 * hits, 0.25)

    def _boost_for_priority_chapters(self, metadata: Dict[str, Any], text: str) -> float:
        """Return a boost in [0, 0.35] for finance/audit/enforcement/dissolution/commission."""
        title_bits = []
        for k in ("title", "section", "chapter", "part"):
            v = metadata.get(k)
            if isinstance(v, str):
                title_bits.append(v)
        title = " ".join(title_bits).lower()
        body = (text or "").lower()
        hay = f"{title} {body[:300]}"

        boost_terms = [
            # Finance
            "finance", "financial", "fund", "budget", "accounts", "accounting", "treasury",
            # Audit
            "audit", "auditor", "audited",
            # Enforcement
            "enforcement", "penalty", "penalties", "offence", "offense", "fines", "prosecution",
            # Dissolution
            "dissolution", "dissolve", "transitional", "transition",
            # Commission (local gov/election commission contexts)
            "commission", "local government commission", "election commission",
        ]
        hits = sum(1 for k in boost_terms if k in hay)
        if hits == 0:
            return 0.0
        # Scale gently, cap boost
        return min(0.06 * hits, 0.35)

    def _apply_domain_bias(self, query: str, result: Dict[str, Any]) -> float:
        """Compute additive bias to apply to the score (positive or negative)."""
        metadata = result.get('metadata', {}) or {}
        text = result.get('text', '') or ''
        bonus = self._boost_for_priority_chapters(metadata, text)
        malus = self._penalty_for_generic(query, metadata, text)
        return bonus - malus

    def _rerank_with_cross_encoder(self, query: str, results: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Return cross-encoder scores aligned to results order, or None if unavailable."""
        if not results or self.reranker is None:
            return None
        pairs = [(query, r.get('text', '') or '') for r in results]
        try:
            scores = self.reranker.predict(pairs)
            if isinstance(scores, list):
                scores = np.asarray(scores, dtype=np.float32)
            return scores
        except Exception as exc:
            print(f"⚠️ Cross-encoder scoring failed: {exc}")
            return None
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        expanded_queries = [query]
        
        # Legal term expansions
        expansions = {
            'council': ['committee', 'board', 'assembly'],
            'government': ['administration', 'authority', 'state'],
            'power': ['authority', 'jurisdiction', 'right'],
            'function': ['duty', 'responsibility', 'role'],
            'election': ['vote', 'ballot', 'polling'],
            'budget': ['fund', 'finance', 'money'],
            'development': ['improvement', 'progress', 'growth'],
            'planning': ['scheme', 'project', 'proposal']
        }
        
        query_lower = query.lower()
        for term, synonyms in expansions.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded_query = query_lower.replace(term, synonym)
                    expanded_queries.append(expanded_query)
        
        # Add question variations
        if '?' in query:
            # Convert question to statement forms
            statement_forms = []
            if query.lower().startswith('what'):
                statement_forms.append(query.replace('what is', '').replace('what are', '').strip(' ?'))
            elif query.lower().startswith('how'):
                statement_forms.append(query.replace('how to', '').replace('how', '').strip(' ?'))
            elif query.lower().startswith('when'):
                statement_forms.append(query.replace('when', 'time').strip(' ?'))
            elif query.lower().startswith('where'):
                statement_forms.append(query.replace('where', 'location').strip(' ?'))
            
            expanded_queries.extend(statement_forms)
        
        return list(set(expanded_queries))
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Perform keyword-based search using BM25"""
        if not self.bm25:
            return []
        
        processed_query = self.preprocess_text(query)
        tokenized_query = processed_query.split()
        
        # Get BM25 scores
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top results
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        results = [(int(idx), float(bm25_scores[idx])) for idx in top_indices if bm25_scores[idx] > 0]
        
        return results
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Perform semantic search using sentence embeddings"""
        if self.document_embeddings is None:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(int(idx), float(similarities[idx])) for idx in top_indices if similarities[idx] > 0.1]
        
        return results
    
    def tfidf_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Perform TF-IDF based search"""
        if self.tfidf_matrix is None:
            return []
        
        processed_query = self.preprocess_text(query)
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(int(idx), float(similarities[idx])) for idx in top_indices if similarities[idx] > 0.05]
        
        return results
    
    def entity_based_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search based on legal entities and references"""
        query_entities = self.extract_legal_entities(query)
        results = []
        
        for i, doc in enumerate(self.documents):
            doc_text = doc.get('text', '')
            doc_entities = self.extract_legal_entities(doc_text)
            metadata = doc.get('metadata', {})
            
            score = 0.0
            
            # Score based on entity matches
            for entity_type in ['sections', 'articles', 'chapters', 'parts', 'schedules', 'clauses', 'subclauses']:
                query_refs = query_entities.get(entity_type, [])
                doc_refs = doc_entities.get(entity_type, [])

                for ref in query_refs:
                    if ref in doc_refs:
                        score += 2.0  # High score for exact reference match
            
            # Score based on legal terms
            query_terms = query_entities.get('legal_terms', [])
            doc_terms = doc_entities.get('legal_terms', [])
            
            common_terms = set(query_terms) & set(doc_terms)
            score += len(common_terms) * 0.5
            
            # Score based on metadata matches
            if metadata:
                for field in ['section_number', 'chapter_number', 'article_number', 'schedule_number']:
                    if field in metadata and metadata[field]:
                        field_type = field.replace('_number', 's')
                        if field_type in query_entities:
                            if str(metadata[field]) in query_entities[field_type]:
                                score += 3.0  # Very high score for metadata match
                if metadata.get('clause_label') and metadata['clause_label'] in query_entities.get('clauses', []):
                    score += 3.0
                if metadata.get('is_clause_variant'):
                    score *= 1.1
            
            if score > 0:
                results.append((i, score))
        
        # Sort by score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def hybrid_search(self, query: str, top_k: int = 5, weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """Perform hybrid search combining multiple methods"""
        if weights is None:
            weights = {
                'semantic': 0.4,
                'keyword': 0.3,
                'tfidf': 0.2,
                'entity': 0.1
            }
        
        # Expand query
        expanded_queries = self.expand_query(query)
        
        all_results = {}
        
        # Perform different types of searches
        for expanded_query in expanded_queries:
            # Semantic search
            semantic_results = self.semantic_search(expanded_query, top_k * 2)
            for idx, score in semantic_results:
                if idx not in all_results:
                    all_results[idx] = {}
                all_results[idx]['semantic'] = max(all_results[idx].get('semantic', 0), score * weights['semantic'])
            
            # Keyword search
            keyword_results = self.keyword_search(expanded_query, top_k * 2)
            for idx, score in keyword_results:
                if idx not in all_results:
                    all_results[idx] = {}
                # Normalize BM25 scores
                normalized_score = min(score / 10.0, 1.0)
                all_results[idx]['keyword'] = max(all_results[idx].get('keyword', 0), normalized_score * weights['keyword'])
            
            # TF-IDF search
            tfidf_results = self.tfidf_search(expanded_query, top_k * 2)
            for idx, score in tfidf_results:
                if idx not in all_results:
                    all_results[idx] = {}
                all_results[idx]['tfidf'] = max(all_results[idx].get('tfidf', 0), score * weights['tfidf'])
        
        # Entity-based search (only on original query)
        entity_results = self.entity_based_search(query, top_k * 2)
        for idx, score in entity_results:
            if idx not in all_results:
                all_results[idx] = {}
            # Normalize entity scores
            normalized_score = min(score / 5.0, 1.0)
            all_results[idx]['entity'] = max(all_results[idx].get('entity', 0), normalized_score * weights['entity'])
        
        # Calculate final scores
        final_results = []
        for idx, scores in all_results.items():
            final_score = sum(scores.values())
            if final_score > 0.1:  # Minimum threshold
                result = {
                    'document_index': idx,
                    'document': self.documents[idx],
                    'score': final_score,
                    'score_breakdown': scores,
                    'text': self.documents[idx].get('text', ''),
                    'metadata': self.documents[idx].get('metadata', {})
                }
                final_results.append(result)
        
        # Domain-aware biasing before re-ranking
        for r in final_results:
            r['score'] = float(r['score'] + self._apply_domain_bias(query, r))

        # Cross-encoder re-ranking: blend with existing score
        # Keep a generous candidate set to reduce drift
        candidate_cap = max(top_k * 6, 30)
        candidates = final_results[:candidate_cap]

        ce_scores = self._rerank_with_cross_encoder(query, candidates)
        if ce_scores is not None:
            # Blend: cross-encoder dominates to reduce noise; retain hybrid as a prior
            alpha = 0.30  # weight for prior hybrid score
            beta = 0.70   # weight for CE score
            # Normalise CE scores to 0..1 per-batch for stability
            ce = ce_scores.astype(np.float32)
            if ce.size > 0:
                cemin = float(np.min(ce))
                cemax = float(np.max(ce))
                if cemax > cemin:
                    ce = (ce - cemin) / (cemax - cemin)
                else:
                    ce = np.zeros_like(ce)
            for i, r in enumerate(candidates):
                prior = float(r.get('score', 0.0))
                r['score'] = alpha * prior + beta * float(ce[i])

        # Final domain-aware touch after CE blending (small adjustments)
        focus_terms = self._extract_focus_terms(query)
        for r in candidates:
            r['score'] = float(r['score'] + 0.25 * self._apply_domain_bias(query, r))
            # Small focus-term boost to privilege contextually tight matches
            ov = self._keyword_overlap_score(focus_terms, r.get('text', '') or '', r.get('metadata', {}) or {})
            r['score'] = float(r['score'] + 0.15 * ov)

        # Sort and return top_k
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_k]
