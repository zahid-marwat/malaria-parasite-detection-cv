"""
Batch diagnosis module for processing multiple blood smear images.

This module provides utilities for batch processing of blood smear images,
aggregating results, and generating summary statistics.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Container for batch processing results."""
    total_images: int
    infected_count: int
    uninfected_count: int
    uncertain_count: int
    processing_time: float
    average_confidence: float
    min_confidence: float
    max_confidence: float
    sensitivity_focus_positive_rate: float
    predictions: List[Dict]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class BatchDiagnoser:
    """
    Batch processing of blood smear images for malaria detection.
    
    Handles:
    - Multiple image processing
    - Result aggregation
    - Statistical analysis
    - Report generation
    
    Clinical Note:
    - Designed for screening large numbers of samples
    - All results must be reviewed by trained microscopists
    - Uncertain cases (20-95% confidence) require expert review
    """
    
    def __init__(self, predictor, batch_size: int = 32,
                 sensitivity_threshold: float = 0.4,
                 specificity_threshold: float = 0.5):
        """
        Initialize batch diagnoser.
        
        Args:
            predictor: MalariaDiagnosticPredictor instance
            batch_size: Number of images to process simultaneously
            sensitivity_threshold: Threshold prioritizing sensitivity (default 0.4)
            specificity_threshold: Standard threshold (default 0.5)
        """
        self.predictor = predictor
        self.batch_size = batch_size
        self.sensitivity_threshold = sensitivity_threshold
        self.specificity_threshold = specificity_threshold
    
    def process_images(self, image_paths: List[str],
                      use_sensitivity_mode: bool = True) -> BatchResult:
        """
        Process a batch of images.
        
        Args:
            image_paths: List of paths to blood smear images
            use_sensitivity_mode: If True, use sensitivity-first thresholding
        
        Returns:
            BatchResult with aggregated statistics
        """
        import time
        start_time = time.time()
        
        predictions = []
        confidences = []
        
        # Process images in batches
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_predictions = self.predictor.predict_batch(batch_paths)
            
            threshold = (self.sensitivity_threshold if use_sensitivity_mode
                        else self.specificity_threshold)
            
            for path, pred in zip(batch_paths, batch_predictions):
                infected_prob = pred['infected_probability']
                confidences.append(max(infected_prob, 1 - infected_prob))
                
                # Classify based on threshold
                if infected_prob > threshold:
                    classification = 'infected'
                elif infected_prob < (1 - threshold):
                    classification = 'uninfected'
                else:
                    classification = 'uncertain'
                
                predictions.append({
                    'image_path': str(path),
                    'classification': classification,
                    'infected_probability': float(infected_prob),
                    'confidence': float(confidences[-1]),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Aggregate results
        infected_count = sum(1 for p in predictions if p['classification'] == 'infected')
        uninfected_count = sum(1 for p in predictions if p['classification'] == 'uninfected')
        uncertain_count = sum(1 for p in predictions if p['classification'] == 'uncertain')
        
        processing_time = time.time() - start_time
        
        result = BatchResult(
            total_images=len(image_paths),
            infected_count=infected_count,
            uninfected_count=uninfected_count,
            uncertain_count=uncertain_count,
            processing_time=processing_time,
            average_confidence=float(np.mean(confidences)),
            min_confidence=float(np.min(confidences)),
            max_confidence=float(np.max(confidences)),
            sensitivity_focus_positive_rate=(infected_count / len(image_paths)
                                           if image_paths else 0),
            predictions=predictions
        )
        
        return result
    
    def process_directory(self, directory: str,
                         extensions: Tuple[str] = ('.jpg', '.jpeg', '.png', '.bmp'),
                         use_sensitivity_mode: bool = True) -> BatchResult:
        """
        Process all images in a directory.
        
        Args:
            directory: Path to directory containing images
            extensions: Tuple of valid image extensions
            use_sensitivity_mode: If True, use sensitivity-first thresholding
        
        Returns:
            BatchResult with all images processed
        """
        directory_path = Path(directory)
        
        # Find all valid images
        image_paths = []
        for ext in extensions:
            image_paths.extend(directory_path.glob(f'*{ext}'))
            image_paths.extend(directory_path.glob(f'*{ext.upper()}'))
        
        image_paths = sorted([str(p) for p in image_paths])
        
        logger.info(f"Found {len(image_paths)} images in {directory}")
        
        return self.process_images(image_paths, use_sensitivity_mode)
    
    def generate_summary_report(self, batch_result: BatchResult) -> str:
        """
        Generate human-readable summary report.
        
        Args:
            batch_result: BatchResult object
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("="*60)
        report.append("BATCH MALARIA SCREENING REPORT")
        report.append("="*60)
        report.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("SUMMARY STATISTICS")
        report.append("-"*60)
        report.append(f"Total images processed: {batch_result.total_images}")
        report.append(f"Processing time: {batch_result.processing_time:.2f} seconds")
        report.append(f"Average time per image: {batch_result.processing_time/batch_result.total_images:.2f}s")
        report.append("")
        
        report.append("CLASSIFICATION RESULTS")
        report.append("-"*60)
        report.append(f"Infected (positive):    {batch_result.infected_count:3d} "
                     f"({100*batch_result.infected_count/batch_result.total_images:.1f}%)")
        report.append(f"Uninfected (negative):  {batch_result.uninfected_count:3d} "
                     f"({100*batch_result.uninfected_count/batch_result.total_images:.1f}%)")
        report.append(f"Uncertain (review):     {batch_result.uncertain_count:3d} "
                     f"({100*batch_result.uncertain_count/batch_result.total_images:.1f}%)")
        report.append("")
        
        report.append("CONFIDENCE STATISTICS")
        report.append("-"*60)
        report.append(f"Average confidence: {batch_result.average_confidence:.1%}")
        report.append(f"Min confidence:     {batch_result.min_confidence:.1%}")
        report.append(f"Max confidence:     {batch_result.max_confidence:.1%}")
        report.append("")
        
        report.append("CLINICAL ACTION ITEMS")
        report.append("-"*60)
        if batch_result.uncertain_count > 0:
            report.append(f"⚠️  {batch_result.uncertain_count} uncertain cases require expert review")
        if batch_result.infected_count > 0:
            report.append(f"✓ {batch_result.infected_count} positive cases detected - recommend clinical confirmation")
        report.append("")
        
        report.append("DISCLAIMER")
        report.append("-"*60)
        report.append("This report is for SCREENING ONLY and must be reviewed by")
        report.append("trained medical professionals. All positive results must be")
        report.append("confirmed by expert microscopy before clinical action.")
        report.append("="*60)
        
        return "\n".join(report)
    
    def export_results_csv(self, batch_result: BatchResult,
                          output_path: str) -> None:
        """
        Export results to CSV file.
        
        Args:
            batch_result: BatchResult object
            output_path: Path to save CSV file
        """
        df = pd.DataFrame(batch_result.predictions)
        df.to_csv(output_path, index=False)
        logger.info(f"Results exported to {output_path}")
    
    def export_results_json(self, batch_result: BatchResult,
                           output_path: str) -> None:
        """
        Export results to JSON file.
        
        Args:
            batch_result: BatchResult object
            output_path: Path to save JSON file
        """
        with open(output_path, 'w') as f:
            f.write(batch_result.to_json())
        logger.info(f"Results exported to {output_path}")
    
    def identify_high_priority_cases(self, batch_result: BatchResult,
                                    uncertainty_range: Tuple[float, float] = (0.3, 0.7)) -> List[Dict]:
        """
        Identify cases requiring urgent expert review.
        
        Cases with predictions near decision boundary are most uncertain.
        
        Args:
            batch_result: BatchResult object
            uncertainty_range: Tuple of (lower, upper) bounds
        
        Returns:
            List of high-priority cases
        """
        high_priority = []
        
        for pred in batch_result.predictions:
            prob = pred['infected_probability']
            
            # Cases near decision boundary
            if uncertainty_range[0] < prob < uncertainty_range[1]:
                high_priority.append(pred)
        
        return sorted(high_priority, key=lambda x: abs(0.5 - x['infected_probability']))
