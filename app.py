"""
Research Paper Authenticity & Plagiarism Detection - Web Application
Main Gradio interface for the complete system
"""

import multiprocessing as mp
import gradio as gr
import os
import json
import tempfile
from pathlib import Path

# Configuration
MODEL_DIR = "./models"
AI_DETECTOR_PATH = os.path.join(MODEL_DIR, "ai_detector_model")
SEMANTIC_INDEX_PATH = os.path.join(MODEL_DIR, "semantic_index")


class AuthenticityDetectionSystem:
    """Main system integrating all components"""

    def __init__(self):
        # NOTE: import heavy/worker modules lazily here to avoid creating child processes
        # before the multiprocessing start method is set in the main guard.
        from src.document_processor import DocumentProcessor
        from src.ai_detector import AIContentDetector
        from src.plagiarism_detector import SemanticPlagiarismDetector
        from src.explainability import ExplainabilityEngine

        print("Initializing Authenticity Detection System...")

        # Initialize components
        self.doc_processor = DocumentProcessor(min_sentence_length=20)
        self.explainer = ExplainabilityEngine()

        # Load models (with error handling)
        self.ai_detector = None
        self.plag_detector = None

        # assign classes for later use (if needed)
        self._AIContentDetector = AIContentDetector
        self._SemanticPlagiarismDetector = SemanticPlagiarismDetector

        self._load_models()

        print("✅ System ready!")

    def _load_models(self):
        """Load trained models"""
        # Load AI detector
        if os.path.exists(AI_DETECTOR_PATH):
            try:
                self.ai_detector = self._AIContentDetector(AI_DETECTOR_PATH)
                print("✅ AI detector loaded")
            except Exception as e:
                print(f"⚠️ Could not load AI detector: {e}")
                print(f"   Place model files in {AI_DETECTOR_PATH}")
        else:
            print(f"⚠️ AI detector not found at {AI_DETECTOR_PATH}")
            print("   Run Colab notebook 01_AI_Content_Detector_Training.ipynb")

        # Load plagiarism detector
        if os.path.exists(SEMANTIC_INDEX_PATH):
            try:
                self.plag_detector = self._SemanticPlagiarismDetector(SEMANTIC_INDEX_PATH)
                print("✅ Plagiarism detector loaded")
            except Exception as e:
                print(f"⚠️ Could not load plagiarism detector: {e}")
                print(f"   Place index files in {SEMANTIC_INDEX_PATH}")
        else:
            print(f"⚠️ Semantic index not found at {SEMANTIC_INDEX_PATH}")
            print("   Run Colab notebook 02_Build_Semantic_Index.ipynb")

    def analyze_document(self, file_obj, progress=gr.Progress()):
        """
        Complete document analysis pipeline

        Args:
            file_obj: Gradio file upload object
            progress: Gradio progress tracker

        Returns:
            Tuple of (summary, annotated_text, detailed_report, score_viz)
        """
        if file_obj is None:
            return "Please upload a document", "", "{}", None

        try:
            progress(0, desc="Processing document...")

            # Process document
            doc_info = self.doc_processor.process_document(file_obj.name)
            sentences = doc_info['sentences']

            if not sentences:
                return "No valid sentences found in document", "", "{}", None

            progress(0.2, desc="Analyzing AI content...")

            # AI Detection
            if self.ai_detector:
                ai_results = self.ai_detector.analyze_document(
                    doc_info['full_text'],
                    sentences,
                    threshold=0.7
                )
            else:
                ai_results = self._mock_ai_results(sentences)

            progress(0.5, desc="Detecting plagiarism...")

            # Plagiarism Detection
            if self.plag_detector:
                plag_results = self.plag_detector.detect(sentences, top_k=5)
            else:
                plag_results = self._mock_plag_results(sentences)

            progress(0.8, desc="Generating report...")

            # Calculate authenticity score
            auth_score = self.explainer.calculate_authenticity_score(
                ai_results, plag_results
            )

            # Generate outputs
            summary = self.explainer.format_summary_text(
                auth_score, ai_results, plag_results
            )

            annotated_text = self.explainer.generate_annotated_text(
                sentences, ai_results, plag_results
            )

            detailed_report = self.explainer.generate_detailed_report(
                doc_info, ai_results, plag_results, auth_score
            )

            # Create visualization data
            viz_data = self._create_visualization_data(auth_score, ai_results, plag_results)

            progress(1.0, desc="Complete!")

            return (
                summary,
                annotated_text,
                json.dumps(detailed_report, indent=2),
                viz_data
            )

        except Exception as e:
            return f"Error analyzing document: {str(e)}", "", "{}", None

    def _mock_ai_results(self, sentences):
        """Mock AI results when model not available"""
        return {
            'overall_assessment': {
                'ai_probability': 0.3,
                'is_likely_ai': False
            },
            'flagged_sentences': [],
            'sentence_level': {
                'total_sentences': len(sentences),
                'ai_percentage': 0
            }
        }

    def _mock_plag_results(self, sentences):
        """Mock plagiarism results when index not available"""
        return {
            'overall_similarity': 0.2,
            'plagiarism_percentage': 0,
            'flagged_sentences': 0,
            'total_sentences': len(sentences),
            'matches': []
        }

    def _create_visualization_data(self, auth_score, ai_results, plag_results):
        """Create data for visualization"""
        import plotly.graph_objects as go

        # Create gauge chart for authenticity score
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=auth_score.overall_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Authenticity Score", 'font': {'size': 24}},
            delta={'reference': 70},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': self._get_score_color(auth_score.overall_score)},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': '#ffebee'},
                    {'range': [40, 70], 'color': '#fff9c4'},
                    {'range': [70, 100], 'color': '#e8f5e9'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="white",
            font={'color': "darkblue", 'family': "Arial"}
        )

        return fig

    def _get_score_color(self, score):
        """Get color based on score"""
        if score >= 70:
            return "#4caf50"  # Green
        elif score >= 40:
            return "#ff9800"  # Orange
        else:
            return "#f44336"  # Red


def create_gradio_interface():
    """Create and configure Gradio interface"""

    # Initialize system
    system = AuthenticityDetectionSystem()

    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    """

    # Create interface
    with gr.Blocks(css=custom_css, title="Research Paper Authenticity Detector") as demo:

        # Header
        gr.HTML("""
        <div class="header">
            <h1>🔍 Research Paper Authenticity & Plagiarism Detection</h1>
            <p>AI-powered system for detecting AI-generated content and semantic plagiarism</p>
        </div>
        """)

        # System status
        status_messages = []
        if system.ai_detector is None:
            status_messages.append("⚠️ AI detector not loaded - please train model using Colab notebook")
        if system.plag_detector is None:
            status_messages.append("⚠️ Plagiarism detector not loaded - please build index using Colab notebook")

        if status_messages:
            gr.HTML(f"""
            <div class="warning-box">
                <b>System Status:</b><br>
                {'<br>'.join(status_messages)}
            </div>
            """)

        # Main interface
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Upload Document")
                file_input = gr.File(
                    label="Upload Research Paper (PDF/TXT)",
                    file_types=[".pdf", ".txt"],
                    type="filepath"
                )

                analyze_btn = gr.Button("🔍 Analyze Document", variant="primary", size="lg")

                gr.Markdown("""
                ### Instructions:
                1. Upload your research paper (PDF or TXT)
                2. Click "Analyze Document"
                3. Review the authenticity report

                ### Detection Features:
                - **AI Content Detection**: Identifies AI-generated text
                - **Semantic Plagiarism**: Finds similar content using meaning
                - **Sentence-level Analysis**: Highlights suspicious passages
                - **Source Attribution**: Links to similar documents
                """)

            with gr.Column(scale=2):
                gr.Markdown("### Analysis Results")

                # Tabs for different outputs
                with gr.Tabs():
                    with gr.Tab("📊 Summary"):
                        summary_output = gr.Markdown(label="Authenticity Report")
                        score_viz = gr.Plot(label="Authenticity Score")

                    with gr.Tab("📝 Annotated Text"):
                        gr.Markdown("""
                        **Legend:**
                        - 🔴 Red: AI-generated + Plagiarized
                        - 🟠 Orange: AI-generated
                        - 🔵 Teal: Similar content found
                        - ⚪ White: No issues detected
                        """)
                        annotated_output = gr.HTML(label="Highlighted Document")

                    with gr.Tab("📄 Detailed Report"):
                        report_output = gr.JSON(label="Complete Analysis Report")

        # Examples section
        gr.Markdown("### Example Documents")
        gr.Examples(
            examples=[],  # Add example file paths if available
            inputs=file_input,
            label="Try these examples"
        )

        # Footer
        gr.Markdown("""
        ---
        ### About This System
        This system combines:
        - **Sentence-BERT embeddings** + **FAISS** for semantic similarity detection
        - **Fine-tuned transformer models** (BERT/RoBERTa) for AI content detection
        - **Explainability layer** for transparent, interpretable results

        **Models trained on:** Google Colab T4 GPU  
        **Technologies:** PyTorch, Transformers, Sentence-Transformers, Gradio
        """)

        # Connect interface
        analyze_btn.click(
            fn=system.analyze_document,
            inputs=[file_input],
            outputs=[summary_output, annotated_output, report_output, score_viz]
        )

    return demo


# Main execution
if __name__ == "__main__":
    # Set spawn start method on macOS before creating any child processes.
    # This must be done in the main guard to avoid RuntimeError.
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # already set by another module; safe to continue
        pass

    # Create interface
    demo = create_gradio_interface()

    # Launch app
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Set to True to create public link
        show_error=True
    )
