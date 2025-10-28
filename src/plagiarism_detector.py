"""
Semantic Plagiarism Detector
Loads pre-built FAISS index from Colab and performs similarity search
"""

import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class PlagiarismMatch:
    """Container for plagiarism match"""
    query_sentence: str
    matched_sentence: str
    similarity_score: float
    source_filename: str
    source_doc_id: int
    sentence_idx: int


class SemanticPlagiarismDetector:
    """Load and use pre-built FAISS index for plagiarism detection"""
    
    def __init__(
        self, 
        index_dir: str,
        similarity_threshold: float = 0.75
    ):
        """
        Initialize detector with pre-built index
        
        Args:
            index_dir: Path to semantic_index directory (from Colab)
            similarity_threshold: Minimum similarity to flag (0-1)
        """
        self.index_dir = index_dir
        self.similarity_threshold = similarity_threshold
        
        print(f"Loading semantic index from {index_dir}")
        
        # Load configuration
        with open(os.path.join(index_dir, 'config.json'), 'r') as f:
            self.config = json.load(f)
        
        print(f"Index contains {self.config['total_sentences']} sentences "
              f"from {self.config['total_documents']} documents")
        
        # Load sentence transformer model
        model_name = self.config['model_name']
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Load FAISS index
        index_path = os.path.join(index_dir, 'faiss_index.bin')
        self.index = faiss.read_index(index_path)
        print(f"FAISS index loaded with {self.index.ntotal} vectors")
        
        # Load reference sentences
        with open(os.path.join(index_dir, 'reference_sentences.pkl'), 'rb') as f:
            self.reference_sentences = pickle.load(f)
        
        # Load metadata
        with open(os.path.join(index_dir, 'reference_metadata.pkl'), 'rb') as f:
            self.reference_metadata = pickle.load(f)
        
        print("Semantic plagiarism detector ready")
    
    def detect(
        self,
        query_sentences: List[str],
        top_k: int = 5
    ) -> Dict:
        """
        Detect plagiarism in query sentences
        
        Args:
            query_sentences: List of sentences to check
            top_k: Number of similar sentences to retrieve per query
            
        Returns:
            Detection results with matches and scores
        """
        if not query_sentences:
            return self._empty_result()
        
        print(f"Analyzing {len(query_sentences)} sentences...")
        
        # Encode query sentences
        query_embeddings = self.model.encode(
            query_sentences,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embeddings)
        
        # Search FAISS index
        similarities, indices = self.index.search(query_embeddings, top_k)
        
        # Process results
        flagged_matches = []
        sentence_scores = []
        
        for q_idx, (query_sent, sims, idxs) in enumerate(
            zip(query_sentences, similarities, indices)
        ):
            max_similarity = float(sims[0])
            sentence_scores.append(max_similarity)
            
            # Flag if above threshold
            if max_similarity >= self.similarity_threshold:
                best_match_idx = idxs[0]
                matched_sentence = self.reference_sentences[best_match_idx]
                metadata = self.reference_metadata[best_match_idx]
                
                match = PlagiarismMatch(
                    query_sentence=query_sent,
                    matched_sentence=matched_sentence,
                    similarity_score=max_similarity,
                    source_filename=metadata['filename'],
                    source_doc_id=metadata['doc_id'],
                    sentence_idx=metadata['sentence_idx']
                )
                flagged_matches.append(match)
        
        # Calculate metrics
        overall_similarity = float(np.mean(sentence_scores))
        plagiarism_percentage = (len(flagged_matches) / len(query_sentences)) * 100
        
        return {
            'overall_similarity': overall_similarity,
            'plagiarism_percentage': plagiarism_percentage,
            'flagged_sentences': len(flagged_matches),
            'total_sentences': len(query_sentences),
            'matches': flagged_matches,
            'sentence_scores': sentence_scores
        }
    
    def get_detailed_matches(
        self,
        sentence: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Get detailed matches for a single sentence
        
        Args:
            sentence: Query sentence
            top_k: Number of matches to return
            
        Returns:
            List of detailed match information
        """
        # Encode sentence
        embedding = self.model.encode([sentence], convert_to_numpy=True)
        faiss.normalize_L2(embedding)
        
        # Search
        similarities, indices = self.index.search(embedding, top_k)
        
        # Format results
        matches = []
        for sim, idx in zip(similarities[0], indices[0]):
            metadata = self.reference_metadata[idx]
            matches.append({
                'matched_text': self.reference_sentences[idx],
                'similarity': float(sim),
                'source_file': metadata['filename'],
                'doc_id': metadata['doc_id'],
                'sentence_idx': metadata['sentence_idx']
            })
        
        return matches
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            'overall_similarity': 0.0,
            'plagiarism_percentage': 0.0,
            'flagged_sentences': 0,
            'total_sentences': 0,
            'matches': [],
            'sentence_scores': []
        }


# Test code
if __name__ == "__main__":
    try:
        detector = SemanticPlagiarismDetector("./models/semantic_index")
        
        test_sentences = [
            "Machine learning has transformed data analysis.",
            "Neural networks are powerful computational models."
        ]
        
        results = detector.detect(test_sentences)
        print(f"Results: {results}")
        
        if results['matches']:
            print(f"\nFound {len(results['matches'])} potential matches")
            for match in results['matches'][:3]:
                print(f"\nQuery: {match.query_sentence[:60]}...")
                print(f"Match: {match.matched_sentence[:60]}...")
                print(f"Similarity: {match.similarity_score:.3f}")
                print(f"Source: {match.source_filename}")
    except Exception as e:
        print(f"Index not found or error: {e}")
        print("Make sure to extract semantic_index.zip to ./models/semantic_index/")