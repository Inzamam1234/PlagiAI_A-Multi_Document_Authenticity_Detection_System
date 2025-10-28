"""
Explainability and Reporting Module
Generate annotated reports with highlighted suspicious sentences
"""

from typing import List, Dict
import numpy as np
from dataclasses import dataclass


@dataclass
class AuthenticityScore:
    """Overall document authenticity assessment"""
    overall_score: float  # 0-100, higher = more authentic
    ai_probability: float  # 0-1, probability of AI generation
    plagiarism_percentage: float  # 0-100, percentage of plagiarized content
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'


class ExplainabilityEngine:
    """Generate comprehensive authenticity reports"""
    
    def __init__(self):
        self.risk_thresholds = {
            'ai_high': 0.7,
            'ai_medium': 0.4,
            'plagiarism_high': 40.0,
            'plagiarism_medium': 20.0
        }
    
    def calculate_authenticity_score(
        self,
        ai_results: Dict,
        plagiarism_results: Dict
    ) -> AuthenticityScore:
        """
        Calculate overall authenticity score
        
        Args:
            ai_results: Results from AI detector
            plagiarism_results: Results from plagiarism detector
            
        Returns:
            AuthenticityScore object
        """
        # Extract key metrics
        ai_prob = ai_results.get('overall_assessment', {}).get('ai_probability', 0)
        plag_pct = plagiarism_results.get('plagiarism_percentage', 0)
        
        # Calculate composite score (0-100, higher = more authentic)
        # Penalize both AI content and plagiarism
        authenticity = 100 - (ai_prob * 50) - (plag_pct * 0.5)
        authenticity = max(0, min(100, authenticity))
        
        # Determine risk level
        risk_level = self._determine_risk_level(ai_prob, plag_pct)
        
        return AuthenticityScore(
            overall_score=float(authenticity),
            ai_probability=float(ai_prob),
            plagiarism_percentage=float(plag_pct),
            risk_level=risk_level
        )
    
    def _determine_risk_level(self, ai_prob: float, plag_pct: float) -> str:
        """Determine overall risk level"""
        if (ai_prob >= self.risk_thresholds['ai_high'] or 
            plag_pct >= self.risk_thresholds['plagiarism_high']):
            return 'HIGH'
        elif (ai_prob >= self.risk_thresholds['ai_medium'] or 
              plag_pct >= self.risk_thresholds['plagiarism_medium']):
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def generate_annotated_text(
        self,
        sentences: List[str],
        ai_results: Dict,
        plagiarism_results: Dict
    ) -> str:
        """
        Generate HTML-formatted annotated text with highlights
        
        Args:
            sentences: Original sentences
            ai_results: AI detection results
            plagiarism_results: Plagiarism detection results
            
        Returns:
            HTML string with highlighted text
        """
        # Build sentence-level annotations
        annotations = self._build_annotations(
            sentences, ai_results, plagiarism_results
        )
        
        # Generate HTML
        html_parts = ['<div style="font-family: Arial, sans-serif; line-height: 1.8;">']
        
        for i, (sent, ann) in enumerate(zip(sentences, annotations)):
            # Determine highlight color
            if ann['is_ai'] and ann['is_plagiarized']:
                color = '#ff6b6b'  # Red - both issues
                label = '⚠️ AI + Plagiarism'
            elif ann['is_ai']:
                color = '#ffa500'  # Orange - AI generated
                label = '🤖 AI-Generated'
            elif ann['is_plagiarized']:
                color = '#4ecdc4'  # Teal - Plagiarized
                label = '📋 Similar Content'
            else:
                color = None
                label = None
            
            # Format sentence
            if color:
                html_parts.append(
                    f'<span style="background-color: {color}; padding: 2px 4px; '
                    f'border-radius: 3px;" title="{label}">'
                    f'{sent}</span> '
                )
            else:
                html_parts.append(f'{sent} ')
        
        html_parts.append('</div>')
        
        return ''.join(html_parts)
    
    def _build_annotations(
        self,
        sentences: List[str],
        ai_results: Dict,
        plagiarism_results: Dict
    ) -> List[Dict]:
        """Build sentence-level annotations"""
        annotations = []
        
        # Get AI flagged sentences
        ai_flagged = set()
        for item in ai_results.get('flagged_sentences', []):
            ai_flagged.add(item['sentence_idx'])
        
        # Get plagiarism flagged sentences
        plag_flagged = {}
        for match in plagiarism_results.get('matches', []):
            # Find sentence index
            for i, sent in enumerate(sentences):
                if sent == match.query_sentence:
                    plag_flagged[i] = {
                        'similarity': match.similarity_score,
                        'source': match.source_filename
                    }
                    break
        
        # Build annotations
        for i in range(len(sentences)):
            annotations.append({
                'is_ai': i in ai_flagged,
                'is_plagiarized': i in plag_flagged,
                'plagiarism_info': plag_flagged.get(i)
            })
        
        return annotations
    
    def generate_detailed_report(
        self,
        doc_info: Dict,
        ai_results: Dict,
        plagiarism_results: Dict,
        authenticity_score: AuthenticityScore
    ) -> Dict:
        """
        Generate comprehensive JSON report
        
        Args:
            doc_info: Document metadata
            ai_results: AI detection results
            plagiarism_results: Plagiarism results
            authenticity_score: Overall score
            
        Returns:
            Complete report dictionary
        """
        # AI Detection Summary
        ai_summary = {
            'overall_probability': ai_results['overall_assessment']['ai_probability'],
            'flagged_sentences': ai_results['flagged_sentences'],
            'total_sentences': ai_results['sentence_level']['total_sentences'],
            'percentage_flagged': ai_results['sentence_level']['ai_percentage']
        }
        
        # Plagiarism Summary
        plag_summary = {
            'overall_similarity': plagiarism_results['overall_similarity'],
            'plagiarism_percentage': plagiarism_results['plagiarism_percentage'],
            'flagged_sentences': plagiarism_results['flagged_sentences'],
            'total_sentences': plagiarism_results['total_sentences']
        }
        
        # Detailed matches
        detailed_matches = []
        for match in plagiarism_results.get('matches', [])[:10]:  # Top 10
            detailed_matches.append({
                'query_text': match.query_sentence[:100] + '...',
                'matched_text': match.matched_sentence[:100] + '...',
                'similarity': match.similarity_score,
                'source_file': match.source_filename
            })
        
        return {
            'document_info': {
                'num_sentences': doc_info['num_sentences'],
                'num_words': doc_info['num_words'],
                'avg_sentence_length': doc_info['avg_sentence_length']
            },
            'authenticity_score': {
                'overall_score': authenticity_score.overall_score,
                'risk_level': authenticity_score.risk_level
            },
            'ai_detection': ai_summary,
            'plagiarism_detection': plag_summary,
            'top_matches': detailed_matches
        }
    
    def format_summary_text(
        self,
        authenticity_score: AuthenticityScore,
        ai_results: Dict,
        plagiarism_results: Dict
    ) -> str:
        """
        Generate human-readable summary
        
        Args:
            authenticity_score: Overall score
            ai_results: AI detection results
            plagiarism_results: Plagiarism results
            
        Returns:
            Formatted summary string
        """
        risk_emoji = {
            'LOW': '✅',
            'MEDIUM': '⚠️',
            'HIGH': '🚨'
        }
        
        summary = f"""
## Authenticity Report

### Overall Assessment
**Authenticity Score:** {authenticity_score.overall_score:.1f}/100
**Risk Level:** {risk_emoji[authenticity_score.risk_level]} {authenticity_score.risk_level}

### AI Content Detection
- **AI Probability:** {authenticity_score.ai_probability:.1%}
- **Flagged Sentences:** {ai_results['flagged_sentences']} / {ai_results['sentence_level']['total_sentences']}
- **Assessment:** {"⚠️ Likely AI-generated" if authenticity_score.ai_probability > 0.7 else "✅ Likely human-written"}

### Plagiarism Detection
- **Overall Similarity:** {plagiarism_results['overall_similarity']:.1%}
- **Plagiarized Content:** {authenticity_score.plagiarism_percentage:.1f}%
- **Flagged Sentences:** {plagiarism_results['flagged_sentences']} / {plagiarism_results['total_sentences']}
- **Assessment:** {"⚠️ Significant similarity detected" if authenticity_score.plagiarism_percentage > 20 else "✅ Minimal similarity"}

### Recommendations
"""
        
        if authenticity_score.risk_level == 'HIGH':
            summary += "- 🚨 **High risk detected** - Manual review strongly recommended\n"
            summary += "- Verify authorship and sources carefully\n"
        elif authenticity_score.risk_level == 'MEDIUM':
            summary += "- ⚠️ **Moderate concerns** - Additional verification suggested\n"
        else:
            summary += "- ✅ **Low risk** - Document appears authentic\n"
        
        return summary


# Test code
if __name__ == "__main__":
    engine = ExplainabilityEngine()
    
    # Mock data
    ai_results = {
        'overall_assessment': {'ai_probability': 0.65},
        'flagged_sentences': [{'sentence_idx': 1}],
        'sentence_level': {'total_sentences': 10, 'ai_percentage': 10}
    }
    
    plagiarism_results = {
        'overall_similarity': 0.55,
        'plagiarism_percentage': 30.0,
        'flagged_sentences': 3,
        'total_sentences': 10,
        'matches': []
    }
    
    score = engine.calculate_authenticity_score(ai_results, plagiarism_results)
    print(f"Authenticity Score: {score.overall_score:.1f}")
    print(f"Risk Level: {score.risk_level}")
    
    summary = engine.format_summary_text(score, ai_results, plagiarism_results)
    print(summary)