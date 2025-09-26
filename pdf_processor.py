import PyPDF2
from typing import List
import re

class PDFProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        
    def extract_text(self) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            
            # If this is not the last chunk, try to break at a sentence or paragraph
            if end < text_length:
                # Look for paragraph break first
                paragraph_break = text.rfind('\n', start, end)
                if paragraph_break > start:
                    end = paragraph_break
                else:
                    # Look for sentence break
                    sentence_break = text.rfind('.', start, end)
                    if sentence_break > start:
                        end = sentence_break + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + chunk_size - overlap, end)
        
        return chunks
    
    def process_pdf(self, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Complete PDF processing pipeline"""
        raw_text = self.extract_text()
        clean_text = self.clean_text(raw_text)
        chunks = self.chunk_text(clean_text, chunk_size, overlap)
        return chunks