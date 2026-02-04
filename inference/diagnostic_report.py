"""
Diagnostic report generation for clinical consultation.

Generates detailed, interpretable reports suitable for clinical review
including visualizations, confidence metrics, and recommendations.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticReport:
    """Clinical diagnostic report for a malaria test."""
    
    patient_id: str
    specimen_id: str
    test_date: str
    image_file: str
    
    # Classification results
    classification: str  # 'infected', 'uninfected', 'uncertain'
    infected_probability: float
    confidence: float
    
    # Clinical metrics
    sensitivity_estimate: float  # Model's estimated sensitivity
    specificity_estimate: float  # Model's estimated specificity
    npv_estimate: float  # Negative predictive value
    ppv_estimate: float  # Positive predictive value
    
    # Interpretation
    interpretation: str
    clinical_recommendation: str
    quality_flags: List[str]
    
    # Metadata
    model_version: str
    threshold_used: float
    reviewer_name: Optional[str] = None
    reviewer_comments: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        from dataclasses import asdict
        return asdict(self)


class DiagnosticReportGenerator:
    """
    Generate clinical-grade diagnostic reports for malaria screening results.
    
    Produces interpretable reports suitable for:
    - Clinical staff review
    - Patient consultation
    - Medical records
    - Quality assurance
    """
    
    def __init__(self, model_version: str = "1.0",
                 institution_name: str = "Malaria Diagnostic AI System"):
        """
        Initialize report generator.
        
        Args:
            model_version: Version of AI model used
            institution_name: Name of institution/system
        """
        self.model_version = model_version
        self.institution_name = institution_name
        
        # Clinical performance estimates (should be updated from validation data)
        self.sensitivity_estimate = 0.95
        self.specificity_estimate = 0.90
        self.npv_estimate = 0.98
        self.ppv_estimate = 0.87
    
    def generate_report(self, prediction: Dict,
                       patient_id: str,
                       specimen_id: str,
                       sensitivity_threshold: float = 0.4) -> DiagnosticReport:
        """
        Generate diagnostic report from prediction.
        
        Args:
            prediction: Dictionary with keys:
                - 'image_path': path to blood smear image
                - 'infected_probability': model output
                - 'confidence': max probability
            patient_id: Patient identifier
            specimen_id: Specimen/slide identifier
            sensitivity_threshold: Threshold used (for interpretation)
        
        Returns:
            DiagnosticReport object
        """
        
        infected_prob = prediction['infected_probability']
        
        # Classify
        if infected_prob > sensitivity_threshold:
            classification = 'infected'
        elif infected_prob < (1 - sensitivity_threshold):
            classification = 'uninfected'
        else:
            classification = 'uncertain'
        
        # Generate interpretation
        interpretation = self._generate_interpretation(
            classification, infected_prob, sensitivity_threshold
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            classification, prediction['confidence']
        )
        
        # Check for quality issues
        quality_flags = self._check_image_quality(prediction)
        
        report = DiagnosticReport(
            patient_id=patient_id,
            specimen_id=specimen_id,
            test_date=datetime.now().isoformat(),
            image_file=prediction['image_path'],
            classification=classification,
            infected_probability=float(infected_prob),
            confidence=float(prediction['confidence']),
            sensitivity_estimate=self.sensitivity_estimate,
            specificity_estimate=self.specificity_estimate,
            npv_estimate=self.npv_estimate,
            ppv_estimate=self.ppv_estimate,
            interpretation=interpretation,
            clinical_recommendation=recommendation,
            quality_flags=quality_flags,
            model_version=self.model_version,
            threshold_used=sensitivity_threshold
        )
        
        return report
    
    def _generate_interpretation(self, classification: str,
                                infected_prob: float,
                                threshold: float) -> str:
        """Generate clinical interpretation text."""
        
        if classification == 'infected':
            return (
                f"Blood smear shows POSITIVE result for malaria parasite.\n"
                f"AI confidence: {infected_prob:.1%}\n"
                f"Interpretation: Parasite likely present in specimen.\n"
                f"CLINICAL ACTION: Expert microscopy confirmation REQUIRED before treatment.\n"
                f"Do not initiate antimalarial therapy based on AI result alone."
            )
        
        elif classification == 'uninfected':
            return (
                f"Blood smear shows NEGATIVE result for malaria parasite.\n"
                f"AI confidence: {(1-infected_prob):.1%}\n"
                f"Interpretation: Parasite not detected.\n"
                f"Estimated NPV: {self.npv_estimate:.1%}\n"
                f"CLINICAL ACTION: If clinical suspicion remains high, consider:\n"
                f"  1. Repeat thick blood smear (parasitemia may be low)\n"
                f"  2. Thin blood smear for species identification\n"
                f"  3. Malaria RDT confirmation"
            )
        
        else:  # uncertain
            return (
                f"Blood smear shows UNCERTAIN result (borderline).\n"
                f"AI confidence: {max(infected_prob, 1-infected_prob):.1%}\n"
                f"Interpretation: Result is close to decision threshold.\n"
                f"CLINICAL ACTION: EXPERT MICROSCOPY REVIEW REQUIRED.\n"
                f"Result should not be used for clinical decision-making\n"
                f"without expert confirmation."
            )
    
    def _generate_recommendation(self, classification: str,
                               confidence: float) -> str:
        """Generate clinical recommendations."""
        
        if classification == 'infected':
            if confidence > 0.95:
                return (
                    "HIGH CONFIDENCE positive result.\n"
                    "Recommend: Expert confirmation, initiate antimalarial treatment protocol.\n"
                    "Consider: Identify parasite species with thin smear for optimal therapy."
                )
            elif confidence > 0.80:
                return (
                    "MODERATE-HIGH positive result.\n"
                    "Recommend: Expert review BEFORE treatment initiation.\n"
                    "Consider: Repeat sampling if expert review negative."
                )
            else:
                return (
                    "LOW-MODERATE positive result.\n"
                    "Recommend: Expert microscopist review REQUIRED.\n"
                    "Consider: Malaria RDT and thick/thin smear combination."
                )
        
        elif classification == 'uninfected':
            if confidence > 0.95:
                return (
                    "HIGH CONFIDENCE negative result.\n"
                    "Recommend: Can reliably rule out malaria if clinical symptoms not present.\n"
                    "If symptoms present despite negative test, consider repeat testing or\n"
                    "alternative diagnosis."
                )
            elif confidence > 0.80:
                return (
                    "MODERATE-HIGH negative result.\n"
                    "Recommend: If clinical suspicion is high, consider repeat microscopy.\n"
                    "May repeat with thin smear for higher sensitivity to low parasitemia."
                )
            else:
                return (
                    "BORDERLINE negative result.\n"
                    "Recommend: Expert review if clinical suspicion of malaria.\n"
                    "Consider: RDT confirmation and repeat testing."
                )
        
        else:  # uncertain
            return (
                "UNCERTAIN result - requires expert review.\n"
                "Recommend: Dedicated microscopy review by experienced microscopist.\n"
                "Alternative: Malaria RDT, thick and thin smear combination for\n"
                "confirmation and species identification."
            )
    
    def _check_image_quality(self, prediction: Dict) -> List[str]:
        """Check for potential image quality issues."""
        flags = []
        
        # This would be enhanced with actual image analysis
        # For now, check metadata if available
        
        if 'image_quality_score' in prediction:
            if prediction['image_quality_score'] < 0.6:
                flags.append("Low image quality - consider repeat smear")
        
        if 'staining_quality' in prediction:
            if prediction['staining_quality'] < 0.5:
                flags.append("Poor staining quality - may affect accuracy")
        
        return flags
    
    def format_report_html(self, report: DiagnosticReport) -> str:
        """Generate HTML format report for clinical review."""
        
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }",
            ".container { max-width: 900px; background: white; padding: 20px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }",
            ".header { border-bottom: 3px solid #2c3e50; padding-bottom: 10px; margin-bottom: 20px; }",
            ".section { margin-bottom: 20px; border-left: 4px solid #3498db; padding-left: 15px; }",
            ".result-positive { border-left-color: #e74c3c; background: #fadbd8; padding: 10px; border-radius: 3px; }",
            ".result-negative { border-left-color: #2ecc71; background: #d5f4e6; padding: 10px; border-radius: 3px; }",
            ".result-uncertain { border-left-color: #f39c12; background: #fdebd0; padding: 10px; border-radius: 3px; }",
            ".confidence { font-size: 24px; font-weight: bold; margin: 10px 0; }",
            ".warning { background: #fff3cd; border: 1px solid #ffc107; padding: 10px; border-radius: 3px; margin: 10px 0; }",
            ".footer { font-size: 10px; color: #7f8c8d; margin-top: 30px; border-top: 1px solid #ecf0f1; padding-top: 10px; }",
            "</style>",
            "</head>",
            "<body>",
            "<div class='container'>",
            f"<div class='header'>",
            f"<h1>{self.institution_name}</h1>",
            f"<h2>Clinical Diagnostic Report: Malaria Screening</h2>",
            f"</div>"
        ]
        
        # Patient info
        html_parts.append("<div class='section'>")
        html_parts.append("<h3>Specimen Information</h3>")
        html_parts.append(f"<p><strong>Patient ID:</strong> {report.patient_id}</p>")
        html_parts.append(f"<p><strong>Specimen ID:</strong> {report.specimen_id}</p>")
        html_parts.append(f"<p><strong>Test Date/Time:</strong> {report.test_date}</p>")
        html_parts.append(f"<p><strong>Image File:</strong> {report.image_file}</p>")
        html_parts.append("</div>")
        
        # Results
        result_class = f"result-{report.classification}"
        html_parts.append(f"<div class='section {result_class}'>")
        html_parts.append(f"<h3>Classification Result</h3>")
        html_parts.append(f"<p><strong>Status:</strong> {report.classification.upper()}</p>")
        html_parts.append(f"<p class='confidence'>Confidence: {report.confidence:.1%}</p>")
        html_parts.append(f"<p>Parasite Probability: {report.infected_probability:.1%}</p>")
        html_parts.append("</div>")
        
        # Interpretation
        html_parts.append("<div class='section'>")
        html_parts.append("<h3>Clinical Interpretation</h3>")
        html_parts.append(f"<p>{report.interpretation.replace(chr(10), '<br>')}</p>")
        html_parts.append("</div>")
        
        # Recommendation
        html_parts.append("<div class='section'>")
        html_parts.append("<h3>Clinical Recommendation</h3>")
        html_parts.append(f"<p>{report.clinical_recommendation.replace(chr(10), '<br>')}</p>")
        html_parts.append("</div>")
        
        # Quality flags
        if report.quality_flags:
            html_parts.append("<div class='warning'>")
            html_parts.append("<h4>⚠️ Quality Flags</h4>")
            html_parts.append("<ul>")
            for flag in report.quality_flags:\n                html_parts.append(f"<li>{flag}</li>")
            html_parts.append("</ul>")
            html_parts.append("</div>")
        
        # Performance metrics
        html_parts.append("<div class='section'>")
        html_parts.append("<h3>Model Performance Estimates</h3>")
        html_parts.append(f"<p><strong>Sensitivity:</strong> {report.sensitivity_estimate:.1%}</p>")
        html_parts.append(f"<p><strong>Specificity:</strong> {report.specificity_estimate:.1%}</p>")
        html_parts.append(f"<p><strong>NPV:</strong> {report.npv_estimate:.1%}</p>")
        html_parts.append(f"<p><strong>PPV:</strong> {report.ppv_estimate:.1%}</p>")
        html_parts.append("</div>")
        
        # Disclaimer
n        html_parts.append("<div class='warning'>")
        html_parts.append("<h4>⚠️ DISCLAIMER</h4>")
        html_parts.append("<p>")
        html_parts.append("This report is generated by an artificial intelligence system ")
        html_parts.append("for SCREENING PURPOSES ONLY. It is NOT a diagnostic tool and ")
        html_parts.append("must NOT be used for clinical decision-making without expert ")
        html_parts.append("microscopy review. All positive results must be confirmed by ")
        html_parts.append("trained medical professionals before any treatment is initiated.")
        html_parts.append("</p>")
        html_parts.append("</div>")
        
        # Footer
        html_parts.append(f"<div class='footer'>")
        html_parts.append(f"<p>Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        html_parts.append(f"<p>Model version: {report.model_version}</p>")
        html_parts.append(f"<p>For research and screening use only. Not FDA approved.</p>")
        html_parts.append(f"</div>")
        
        html_parts.append("</div>")
        html_parts.append("</body>")
        html_parts.append("</html>")
        
        return "\n".join(html_parts)
    
    def format_report_text(self, report: DiagnosticReport) -> str:
        """Generate plain text format report."""
        
        text_parts = [
            "="*70,
            f"{self.institution_name}",
            "CLINICAL DIAGNOSTIC REPORT: MALARIA SCREENING",
            "="*70,
            "",
            "SPECIMEN INFORMATION:",
            f"  Patient ID:     {report.patient_id}",
            f"  Specimen ID:    {report.specimen_id}",
            f"  Test Date/Time: {report.test_date}",
            f"  Image File:     {report.image_file}",
            "",
            "CLASSIFICATION RESULT:",
            f"  Status:             {report.classification.upper()}",
            f"  Confidence:         {report.confidence:.1%}",
            f"  Parasite Probability: {report.infected_probability:.1%}",
            "",
            "INTERPRETATION:",
            report.interpretation,
            "",
            "CLINICAL RECOMMENDATION:",
            report.clinical_recommendation,
            "",
            "MODEL PERFORMANCE ESTIMATES:",
            f"  Sensitivity: {report.sensitivity_estimate:.1%}",
            f"  Specificity: {report.specificity_estimate:.1%}",
            f"  NPV:        {report.npv_estimate:.1%}",
            f"  PPV:        {report.ppv_estimate:.1%}",
            "",
            "DISCLAIMER:",
            "This report is for screening purposes only and must NOT be used",
            "for clinical decision-making without expert microscopy review.",
            "All positive results must be confirmed by trained professionals.",
            "="*70,
        ]
        
        return "\n".join(text_parts)
    
    def save_report_json(self, report: DiagnosticReport,
                        output_path: str) -> None:
        """Save report as JSON."""
        import json
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Report saved to {output_path}")
    
    def save_report_html(self, report: DiagnosticReport,
                        output_path: str) -> None:
        """Save report as HTML."""
        html = self.format_report_html(report)
        with open(output_path, 'w') as f:
            f.write(html)
        logger.info(f"Report saved to {output_path}")
    
    def save_report_text(self, report: DiagnosticReport,
                        output_path: str) -> None:
        """Save report as plain text."""
        text = self.format_report_text(report)
        with open(output_path, 'w') as f:
            f.write(text)
        logger.info(f"Report saved to {output_path}")
