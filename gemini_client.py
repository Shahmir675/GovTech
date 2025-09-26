import os
import google.generativeai as genai
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class GeminiClient:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def generate_response(self, prompt: str, context_documents: List[str] = None) -> str:
        """Generate response using Gemini with RAG context"""
        try:
            # Prepare the prompt with context
            if context_documents:
                context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(context_documents)])
                
                full_prompt = f"""You are an AI assistant specialized in the Khyber Pakhtunkhwa Local Government Act 2013. 
Use the following context documents to answer the user's question accurately and comprehensively.

Context Documents:
{context}

User Question: {prompt}

Instructions:
- Base your answer primarily on the provided context documents
- If the context doesn't contain enough information, clearly state that
- Provide specific references to relevant sections or clauses when possible
- Be accurate and helpful in your response
- If asked about specific legal provisions, quote them directly when relevant

Answer:"""
            else:
                full_prompt = f"""You are an AI assistant specialized in the Khyber Pakhtunkhwa Local Government Act 2013. 
Please answer the following question:

{prompt}

Note: No specific context documents were provided for this query."""
            
            response = self.model.generate_content(full_prompt)
            return response.text
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def summarize_document(self, text: str, max_length: int = 500) -> str:
        """Summarize a document or text chunk"""
        try:
            prompt = f"""Please provide a concise summary of the following text from the Khyber Pakhtunkhwa Local Government Act 2013. 
Keep the summary under {max_length} words and focus on key legal provisions and important points.

Text to summarize:
{text}

Summary:"""
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Error summarizing document: {str(e)}"
    
    def extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics from a text"""
        try:
            prompt = f"""Extract the main legal topics and subjects covered in the following text from the Khyber Pakhtunkhwa Local Government Act 2013. 
Return them as a simple list of topics, one per line.

Text:
{text}

Key Topics:"""
            
            response = self.model.generate_content(prompt)
            topics = [topic.strip() for topic in response.text.split('\n') if topic.strip()]
            return topics
            
        except Exception as e:
            return [f"Error extracting topics: {str(e)}"]
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """Validate and categorize user query"""
        try:
            prompt = f"""Analyze the following user query about the Khyber Pakhtunkhwa Local Government Act 2013:

Query: "{query}"

Determine:
1. Is this a valid legal query? (Yes/No)
2. What category does it fall into? (general_info, specific_provision, procedural, definitions, other)
3. What is the main intent? (brief description)
4. Are there any specific legal terms or concepts mentioned?

Respond in this format:
Valid: [Yes/No]
Category: [category]
Intent: [brief description]
Legal Terms: [comma-separated list or "None"]"""

            response = self.model.generate_content(prompt)
            return {"analysis": response.text, "is_valid": True}
            
        except Exception as e:
            return {"analysis": f"Error analyzing query: {str(e)}", "is_valid": False}