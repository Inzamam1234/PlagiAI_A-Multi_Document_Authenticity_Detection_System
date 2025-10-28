"""
AI Content Detector
Loads trained model from Colab and provides inference interface
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import List, Dict
import numpy as np


class AIContentDetector:
    """Detect AI-generated content using fine-tuned transformer model"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize AI detector with trained model
        
        Args:
            model_path: Path to trained model directory (from Colab)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        print(f"Loading AI detector from {model_path}")
        
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Create pipeline for easy inference
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == 'cuda' else -1
        )
        
        print(f"Model loaded on {self.device}")
    
    def predict(self, text: str) -> Dict:
        """
        Predict if text is AI-generated
        
        Args:
            text: Input text
            
        Returns:
            Dict with prediction and confidence
        """
        result = self.classifier(text, truncation=True, max_length=512)[0]
        
        # Parse label (LABEL_0 = human, LABEL_1 = AI)
        is_ai = result['label'] == 'LABEL_1'
        confidence = result['score']
        
        return {
            'is_ai_generated': is_ai,
            'confidence': confidence,
            'label': 'AI-Generated' if is_ai else 'Human-Written'
        }
    
    def predict_sentences(self, sentences: List[str]) -> Dict:
        """
        Predict AI generation for multiple sentences
        
        Args:
            sentences: List of sentences
            
        Returns:
            Dict with per-sentence predictions and overall score
        """
        predictions = []
        ai_scores = []
        
        for sentence in sentences:
            pred = self.predict(sentence)
            predictions.append(pred)
            
            # Store probability of being AI (higher = more likely AI)
            ai_prob = pred['confidence'] if pred['is_ai_generated'] else (1 - pred['confidence'])
            ai_scores.append(ai_prob)
        
        # Calculate overall AI probability
        overall_ai_score = np.mean(ai_scores)
        ai_sentence_count = sum(1 for p in predictions if p['is_ai_generated'])
        ai_percentage = (ai_sentence_count / len(sentences)) * 100 if sentences else 0
        
        return {
            'overall_ai_score': float(overall_ai_score),
            'ai_percentage': float(ai_percentage),
            'flagged_sentences': ai_sentence_count,
            'total_sentences': len(sentences),
            'sentence_predictions': predictions,
            'sentence_scores': ai_scores
        }
    
    def analyze_document(
        self, 
        full_text: str, 
        sentences: List[str],
        threshold: float = 0.7
    ) -> Dict:
        """
        Full document analysis with sentence-level details
        
        Args:
            full_text: Complete document text
            sentences: List of sentences
            threshold: Confidence threshold for flagging
            
        Returns:
            Comprehensive analysis results
        """
        # Analyze full document
        doc_prediction = self.predict(full_text[:512])  # First 512 tokens
        
        # Analyze sentences
        sentence_analysis = self.predict_sentences(sentences)
        
        # Flag high-confidence AI sentences
        flagged_sentences = []
        for i, (sent, pred, score) in enumerate(zip(
            sentences, 
            sentence_analysis['sentence_predictions'],
            sentence_analysis['sentence_scores']
        )):
            if pred['is_ai_generated'] and pred['confidence'] >= threshold:
                flagged_sentences.append({
                    'sentence_idx': i,
                    'text': sent,
                    'confidence': pred['confidence'],
                    'ai_score': score
                })
        
        return {
            'document_level': doc_prediction,
            'sentence_level': sentence_analysis,
            'flagged_sentences': flagged_sentences,
            'overall_assessment': {
                'is_likely_ai': sentence_analysis['overall_ai_score'] > 0.5,
                'ai_probability': sentence_analysis['overall_ai_score'],
                'ai_sentence_percentage': sentence_analysis['ai_percentage']
            }
        }


if __name__ == "__main__":
    # Test code
    try:
        detector = AIContentDetector("./models/ai_detector")
        
        test_texts = [
            "The experimental methodology employed rigorous statistical analysis.",
            "This paper explores various innovative solutions to modern challenges."
        ]
        
        for text in test_texts:
            result = detector.predict(text)
            print(f"Text: {text}")
            print(f"Result: {result}\n")
    except Exception as e:
        print(f"Model not found or error: {e}")
        print("Make sure to extract ai_detector_model.zip to ./models/ai_detector/")