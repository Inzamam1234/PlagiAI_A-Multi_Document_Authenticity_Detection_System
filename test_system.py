"""
System Testing Script
Test all components independently and integrated
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_document_processor():
    """Test document processing"""
    print("\n=== Testing Document Processor ===")
    try:
        from src.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        # Create test document
        test_text = """
        Introduction
        
        Machine learning has revolutionized the field of artificial intelligence.
        Neural networks can learn complex patterns from data.
        This research explores novel approaches to deep learning.
        
        Methodology
        
        We employed a transformer-based architecture for our experiments.
        The dataset consisted of 10,000 samples collected over six months.
        """
        
        # Test text processing
        result = processor.extract_from_content(test_text)
        
        print(f"✓ Processed {result['num_sentences']} sentences")
        print(f"✓ Word count: {result['num_words']}")
        print(f"✓ First sentence: {result['sentences'][0][:60]}...")
        
        return True
    except Exception as e:
        print(f"✗ Plagiarism detector test failed: {e}")
        return False


def test_explainability():
    """Test explainability engine"""
    print("\n=== Testing Explainability Engine ===")
    try:
        from src.explainability import ExplainabilityEngine
        
        engine = ExplainabilityEngine()
        
        # Mock results
        ai_results = {
            'overall_assessment': {'ai_probability': 0.45},
            'flagged_sentences': [],
            'sentence_level': {'total_sentences': 5, 'ai_percentage': 20}
        }
        
        plag_results = {
            'overall_similarity': 0.35,
            'plagiarism_percentage': 15.0,
            'flagged_sentences': 1,
            'total_sentences': 5,
            'matches': []
        }
        
        # Test score calculation
        score = engine.calculate_authenticity_score(ai_results, plag_results)
        
        print(f"✓ Explainability engine initialized")
        print(f"✓ Authenticity score: {score.overall_score:.1f}/100")
        print(f"✓ Risk level: {score.risk_level}")
        
        # Test report generation
        summary = engine.format_summary_text(score, ai_results, plag_results)
        print(f"✓ Report generated successfully")
        
        return True
    except Exception as e:
        print(f"✗ Explainability test failed: {e}")
        return False


def test_integrated_system():
    """Test integrated system"""
    print("\n=== Testing Integrated System ===")
    try:
        from src.document_processor import DocumentProcessor
        from src.ai_detector import AIContentDetector
        from src.plagiarism_detector import SemanticPlagiarismDetector
        from src.explainability import ExplainabilityEngine
        
        # Check if models exist
        ai_model_exists = os.path.exists("./models/ai_detector/config.json")
        plag_model_exists = os.path.exists("./models/semantic_index/config.json")
        
        if not (ai_model_exists and plag_model_exists):
            print("⚠ Cannot test integration - models not available")
            print(f"  AI Detector: {'✓' if ai_model_exists else '✗'}")
            print(f"  Plagiarism Detector: {'✓' if plag_model_exists else '✗'}")
            return False
        
        # Initialize components
        processor = DocumentProcessor()
        ai_detector = AIContentDetector("./models/ai_detector")
        plag_detector = SemanticPlagiarismDetector("./models/semantic_index")
        explainer = ExplainabilityEngine()
        
        # Test document
        test_text = """
        Abstract
        
        This paper presents a comprehensive analysis of machine learning techniques.
        We explore various neural network architectures and their applications.
        The experimental results demonstrate significant improvements in accuracy.
        Deep learning models have shown remarkable performance in classification tasks.
        Our approach combines multiple techniques to achieve state-of-the-art results.
        """
        
        # Process document
        doc_info = processor.extract_from_content(test_text)
        sentences = doc_info['sentences']
        
        # AI detection
        ai_results = ai_detector.analyze_document(doc_info['full_text'], sentences)
        
        # Plagiarism detection
        plag_results = plag_detector.detect(sentences)
        
        # Generate report
        auth_score = explainer.calculate_authenticity_score(ai_results, plag_results)
        
        print(f"✓ Integrated system test passed")
        print(f"✓ Processed {len(sentences)} sentences")
        print(f"✓ AI probability: {auth_score.ai_probability:.2%}")
        print(f"✓ Plagiarism: {auth_score.plagiarism_percentage:.1f}%")
        print(f"✓ Final score: {auth_score.overall_score:.1f}/100")
        
        return True
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("Research Paper Authenticity Detection System")
    print("Component Testing")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Document Processor", test_document_processor()))
    results.append(("AI Detector", test_ai_detector()))
    results.append(("Plagiarism Detector", test_plagiarism_detector()))
    results.append(("Explainability Engine", test_explainability()))
    results.append(("Integrated System", test_integrated_system()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for component, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{component:.<40} {status}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! System is ready.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please check configuration.")
        return 1