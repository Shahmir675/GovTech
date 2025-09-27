import google.generativeai as genai
from typing import List, Dict, Any

class GeminiClient:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate response using Gemini with RAG context"""
        
        # Prepare context from retrieved documents
        context_text = "\n\n".join([
            f"Document {i+1} (Score: {doc['score']:.3f}):\n{doc['text']}"
            for i, doc in enumerate(context)
        ])
        
        # Create prompt template
        prompt = f"""You are an AI assistant specialized in the Khyber Pakhtunkhwa Local Government Act 2013. 
Use the provided context to answer questions accurately and comprehensively.

CONTEXT:
{context_text}

QUESTION: {query}

INSTRUCTIONS:
- Base your answer primarily on the provided context
- If the context doesn't contain sufficient information, clearly state this limitation
- Provide specific references to sections, chapters, or articles when available
- Be precise and cite relevant legal provisions
- If asked about definitions, procedures, or specific requirements, provide detailed explanations
- Maintain a helpful and professional tone

ANSWER:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
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