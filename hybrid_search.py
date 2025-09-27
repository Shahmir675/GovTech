from typing import List, Dict, Any, Tuple, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

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
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            max_df=0.85,
            min_df=2
        )
        self.tfidf_matrix = None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Legal-specific stop words and terms
        self.legal_stop_words = {
            'shall', 'may', 'must', 'should', 'will', 'can', 'could',
            'would', 'might', 'said', 'aforesaid', 'thereof', 'wherein',
            'whereas', 'hereby', 'herein', 'hereafter', 'heretofore'
        }
        
        self.documents = []
        self.document_embeddings = None
        self.processed_docs = []
        
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
        tokens = word_tokenize(text)
        
        # Remove stop words and apply stemming
        processed_tokens = []
        for token in tokens:
            if (token not in self.stop_words and 
                token not in self.legal_stop_words and
                len(token) > 2):
                stemmed = self.stemmer.stem(token)
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

        clauses = re.findall(r'clause\s+(\d+|[a-z])', text.lower())
        entities['clauses'] = list(set(clauses))

        subclauses = re.findall(r'sub-?clause\s+(\d+|[a-z]|[ivxlcdm]+)', text.lower())
        entities['subclauses'] = list(set(subclauses))

        # Extract legal terms
        legal_patterns = [
            r'\b(?:council|committee|government|district|union|local)\b',
            r'\b(?:power|function|duty|responsibility|authority)\b',
            r'\b(?:election|member|chairman|mayor|administrator)\b',
            r'\b(?:budget|fund|tax|fee|development|planning)\b',
            r'\b(?:notification|rule|regulation|ordinance|act)\b'
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
            for entity_type in ['sections', 'articles', 'chapters', 'parts', 'clauses', 'subclauses']:
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
                for field in ['section_number', 'chapter_number', 'article_number']:
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
        
        # Sort by final score
        final_results.sort(key=lambda x: x['score'], reverse=True)
        
        return final_results[:top_k]
