"""
Document Processing Module
Extract and preprocess text from PDF and TXT files
"""

import re
from typing import List, Dict
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class DocumentProcessor:
    """Extract and preprocess research papers"""
    
    def __init__(self, min_sentence_length: int = 20):
        """
        Initialize processor
        
        Args:
            min_sentence_length: Minimum character length for valid sentences
        """
        self.min_sentence_length = min_sentence_length
    
    def extract_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted and cleaned text
        """
        try:
            text = extract_text(pdf_path)
            return self._clean_text(text)
        except (PDFSyntaxError, FileNotFoundError) as e:
            raise ValueError(f"Error extracting PDF: {str(e)}")
    
    def extract_from_txt(self, txt_path: str) -> str:
        """
        Extract text from TXT file
        
        Args:
            txt_path: Path to TXT file
            
        Returns:
            Cleaned text content
        """
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return self._clean_text(text)
        except FileNotFoundError as e:
            raise ValueError(f"Error reading TXT file: {str(e)}")
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(txt_path, 'r', encoding='latin-1') as f:
                    text = f.read()
                return self._clean_text(text)
            except Exception as e:
                raise ValueError(f"Error decoding file: {str(e)}")
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text based on file extension
        
        Args:
            file_path: Path to document
            
        Returns:
            Extracted and cleaned text
        """
        if file_path.lower().endswith('.pdf'):
            return self.extract_from_pdf(file_path)
        elif file_path.lower().endswith('.txt'):
            return self.extract_from_txt(file_path)
        else:
            raise ValueError("Unsupported file format. Use PDF or TXT files.")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and headers/footers patterns
        text = re.sub(r'\n\d+\n', ' ', text)
        # Keep alphanumeric, punctuation, and common symbols
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\']+', '', text)
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences with filtering
        
        Args:
            text: Input text
            
        Returns:
            List of cleaned sentences
        """
        sentences = sent_tokenize(text)
        
        # Filter and clean sentences
        filtered = []
        for sent in sentences:
            sent = sent.strip()
            # Skip very short sentences and those that look like artifacts
            if (len(sent) >= self.min_sentence_length and 
                not self._is_artifact(sent)):
                filtered.append(sent)
        
        return filtered
    
    def _is_artifact(self, sentence: str) -> bool:
        """
        Check if sentence is likely a header/footer/artifact
        
        Args:
            sentence: Sentence to check
            
        Returns:
            True if likely an artifact
        """
        # Check for common artifacts
        artifacts = [
            sentence.isdigit(),  # Just a number
            sentence.isupper() and len(sentence) < 50,  # ALL CAPS header
            sentence.startswith('Figure '),
            sentence.startswith('Table '),
            sentence.startswith('www.'),
            '@' in sentence and len(sentence) < 50,  # Email
        ]
        return any(artifacts)
    
    def process_document(self, file_path: str) -> Dict:
        """
        Complete document processing pipeline
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dictionary with processed document data
        """
        # Extract text
        full_text = self.extract_text(file_path)
        
        # Split into sentences
        sentences = self.split_into_sentences(full_text)
        
        # Calculate statistics
        word_count = len(full_text.split())
        char_count = len(full_text)
        
        return {
            'full_text': full_text,
            'sentences': sentences,
            'num_sentences': len(sentences),
            'num_words': word_count,
            'num_characters': char_count,
            'avg_sentence_length': word_count / len(sentences) if sentences else 0
        }
    
    def extract_from_content(self, content: str, file_type: str = 'txt') -> Dict:
        """
        Process text content directly (for Gradio file upload)
        
        Args:
            content: Text content
            file_type: 'txt' or 'pdf'
            
        Returns:
            Processed document data
        """
        # Clean and split
        cleaned_text = self._clean_text(content)
        sentences = self.split_into_sentences(cleaned_text)
        
        word_count = len(cleaned_text.split())
        
        return {
            'full_text': cleaned_text,
            'sentences': sentences,
            'num_sentences': len(sentences),
            'num_words': word_count,
            'num_characters': len(cleaned_text),
            'avg_sentence_length': word_count / len(sentences) if sentences else 0
        }


if __name__ == "__main__":
    # Test code
    processor = DocumentProcessor()
    
    # Test with sample text
    sample_text = """
    Research Paper Title
    
    Abstract: This paper presents a novel approach to machine learning. 
    The methodology employs advanced techniques for data analysis.
    Results demonstrate significant improvements over baseline methods.
    
    Introduction
    Machine learning has revolutionized data science. Our approach builds 
    on previous work while introducing innovative solutions.
    """
    
    result = processor.extract_from_content(sample_text)
    print(f"Processed document:")
    print(f"  Sentences: {result['num_sentences']}")
    print(f"  Words: {result['num_words']}")
    print(f"  First sentence: {result['sentences'][0][:60]}...")