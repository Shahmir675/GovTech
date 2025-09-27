import google.generativeai as genai
from typing import List, Dict, Any, Tuple
import re
import json

class EnhancedGeminiClient:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Legal document analysis templates
        self.legal_analysis_prompts = {
            'definition': """You are a legal expert analyzing the Khyber Pakhtunkhwa Local Government Act 2013. 
            Provide precise definitions and explanations based on the provided context. Always cite specific sections, articles, or clauses when available.""",
            
            'process': """You are a legal expert explaining procedures and processes from the Khyber Pakhtunkhwa Local Government Act 2013. 
            Break down complex processes into clear, step-by-step instructions. Reference specific legal provisions.""",
            
            'authority': """You are a legal expert explaining powers, authorities, and jurisdictions under the Khyber Pakhtunkhwa Local Government Act 2013. 
            Clearly delineate who has what authority and under which circumstances.""",
            
            'requirements': """You are a legal expert explaining requirements, conditions, and compliance matters under the Khyber Pakhtunkhwa Local Government Act 2013. 
            List all requirements clearly and specify any conditions or exceptions."""
        }
        
    def analyze_query_type(self, query: str) -> str:
        """Analyze query to determine the most appropriate response type"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what is', 'define', 'definition', 'meaning']):
            return 'definition'
        elif any(word in query_lower for word in ['how to', 'process', 'procedure', 'steps', 'method']):
            return 'process'
        elif any(word in query_lower for word in ['power', 'authority', 'jurisdiction', 'responsibility', 'who can']):
            return 'authority'
        elif any(word in query_lower for word in ['requirement', 'condition', 'must', 'shall', 'need to']):
            return 'requirements'
        else:
            return 'definition'  # Default
    
    def extract_citations(self, context: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Extract and organize citations from context metadata"""
        citations = {
            'sections': [],
            'chapters': [],
            'articles': [],
            'parts': []
        }
        
        for doc in context:
            metadata = doc.get('metadata', {})
            
            # Extract structural references
            if 'section_number' in metadata and metadata['section_number']:
                section_ref = f"Section {metadata['section_number']}"
                if 'section' in metadata and metadata['section']:
                    section_ref += f": {metadata['section']}"
                citations['sections'].append(section_ref)
            
            if 'chapter_number' in metadata and metadata['chapter_number']:
                chapter_ref = f"Chapter {metadata['chapter_number']}"
                if 'chapter' in metadata and metadata['chapter']:
                    chapter_ref += f": {metadata['chapter']}"
                citations['chapters'].append(chapter_ref)
            
            if 'article_number' in metadata and metadata['article_number']:
                article_ref = f"Article {metadata['article_number']}"
                if 'article' in metadata and metadata['article']:
                    article_ref += f": {metadata['article']}"
                citations['articles'].append(article_ref)
            
            if 'part_number' in metadata and metadata['part_number']:
                part_ref = f"Part {metadata['part_number']}"
                if 'part' in metadata and metadata['part']:
                    part_ref += f": {metadata['part']}"
                citations['parts'].append(part_ref)
        
        # Remove duplicates while preserving order
        for key in citations:
            citations[key] = list(dict.fromkeys(citations[key]))
        
        return citations
    
    def create_enhanced_context(self, context: List[Dict[str, Any]]) -> Tuple[str, Dict]:
        """Create enhanced context with metadata and citations"""
        context_parts = []
        all_metadata = []
        
        for i, doc in enumerate(context):
            metadata = doc.get('metadata', {})
            text = doc.get('text', '')
            score = doc.get('score', 0)
            
            # Create structured context entry
            context_entry = f"SOURCE {i+1} (Relevance: {score:.3f})"
            
            # Add metadata information
            if metadata:
                meta_parts = []
                if metadata.get('chapter'):
                    meta_parts.append(f"Chapter: {metadata['chapter']}")
                if metadata.get('section'):
                    meta_parts.append(f"Section: {metadata['section']}")
                if metadata.get('article'):
                    meta_parts.append(f"Article: {metadata['article']}")
                
                if meta_parts:
                    context_entry += f" - {', '.join(meta_parts)}"
            
            context_entry += f"\n{text}\n"
            context_parts.append(context_entry)
            all_metadata.append(metadata)
        
        context_text = "\n".join(context_parts)
        citations = self.extract_citations(context)
        
        return context_text, citations
    
    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate enhanced response using Gemini with advanced RAG context"""
        
        if not context:
            return "I don't have sufficient information from the KPK Local Government Act 2013 to answer your question. Please try rephrasing your query or ask about specific topics covered in the act."
        
        # Analyze query type for appropriate prompting
        query_type = self.analyze_query_type(query)
        base_prompt = self.legal_analysis_prompts.get(query_type, self.legal_analysis_prompts['definition'])
        
        # Create enhanced context with metadata
        context_text, citations = self.create_enhanced_context(context)
        
        # Build comprehensive prompt
        prompt = f"""{base_prompt}

RELEVANT SOURCES FROM KPK LOCAL GOVERNMENT ACT 2013:
{context_text}

USER QUESTION: {query}

RESPONSE REQUIREMENTS:
1. **Accuracy**: Base your answer strictly on the provided sources
2. **Citations**: Reference specific sections, articles, or parts when mentioned in the sources
3. **Completeness**: Address all aspects of the question using available information
4. **Structure**: Organize your response with clear headings if covering multiple points
5. **Limitations**: If the sources don't fully answer the question, state what additional information would be needed
6. **Context**: Explain how different provisions relate to each other when relevant

RESPONSE FORMAT:
- Start with a direct answer to the question
- Provide detailed explanation based on the sources
- Include relevant citations in format: (Source X, Section/Article Y)
- End with any important caveats or related information

ANSWER:"""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Post-process response to add citation summary
            citation_summary = self.create_citation_summary(citations)
            if citation_summary:
                response_text += f"\n\n**LEGAL REFERENCES:**\n{citation_summary}"
            
            return response_text
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def create_citation_summary(self, citations: Dict[str, List[str]]) -> str:
        """Create a summary of legal citations"""
        summary_parts = []
        
        for citation_type, refs in citations.items():
            if refs:
                type_name = citation_type.capitalize()
                summary_parts.append(f"• **{type_name}**: {', '.join(refs)}")
        
        return '\n'.join(summary_parts) if summary_parts else ""
    
    def generate_detailed_analysis(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed analysis with structured output"""
        response_text = self.generate_response(query, context)
        
        # Extract key information
        analysis = {
            'response': response_text,
            'query_type': self.analyze_query_type(query),
            'citations': self.extract_citations(context),
            'context_quality': {
                'num_sources': len(context),
                'avg_relevance': sum(doc.get('score', 0) for doc in context) / len(context) if context else 0,
                'has_metadata': any(doc.get('metadata') for doc in context)
            },
            'coverage_analysis': self.analyze_coverage(query, context)
        }
        
        return analysis
    
    def analyze_coverage(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how well the context covers the query"""
        query_terms = set(query.lower().split())
        
        coverage_stats = {
            'query_terms_found': 0,
            'total_query_terms': len(query_terms),
            'sources_with_matches': 0,
            'key_terms_coverage': {}
        }
        
        key_legal_terms = [
            'government', 'council', 'committee', 'district', 'union',
            'power', 'function', 'duty', 'authority', 'election',
            'budget', 'planning', 'development', 'service'
        ]
        
        for term in key_legal_terms:
            if term in query.lower():
                coverage_stats['key_terms_coverage'][term] = 0
                for doc in context:
                    if term in doc.get('text', '').lower():
                        coverage_stats['key_terms_coverage'][term] += 1
        
        # Count query terms found in context
        found_terms = set()
        sources_with_matches = 0
        
        for doc in context:
            doc_text = doc.get('text', '').lower()
            doc_has_match = False
            
            for term in query_terms:
                if term in doc_text:
                    found_terms.add(term)
                    doc_has_match = True
            
            if doc_has_match:
                sources_with_matches += 1
        
        coverage_stats['query_terms_found'] = len(found_terms)
        coverage_stats['sources_with_matches'] = sources_with_matches
        coverage_stats['coverage_percentage'] = (len(found_terms) / len(query_terms)) * 100 if query_terms else 0
        
        return coverage_stats
    
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate a summary of the given text"""
        prompt = f"""Please provide a concise summary of the following text in approximately {max_length} words:

{text}

Summary:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics from the text"""
        prompt = f"""Extract the main topics and themes from this legal document. 
Provide a bulleted list of key topics covered:

{text}

Key Topics:"""
        
        try:
            response = self.model.generate_content(prompt)
            # Parse the response to extract topics
            topics = []
            for line in response.text.split('\n'):
                if line.strip().startswith('-') or line.strip().startswith('•'):
                    topics.append(line.strip()[1:].strip())
            return topics
        except Exception as e:
            return [f"Error extracting topics: {str(e)}"]

# For backward compatibility
GeminiClient = EnhancedGeminiClient