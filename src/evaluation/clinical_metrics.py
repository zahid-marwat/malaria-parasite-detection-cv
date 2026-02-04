"""
Clinical evaluation metrics for medical malaria detection.

Computes medical-appropriate metrics with focus on:
- Sensitivity (Recall): % infected cases correctly identified
- Specificity: % uninfected cases correctly identified
- NPV: Negative Predictive Value - confidence in negative predictions
- These metrics prioritize not missing infections (high sensitivity).
"""

import logging
from typing import Dict, Tuple, Optional, Union

import numpy as np
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, f2_score,
    accuracy_score, roc_auc_score, auc, roc_curve, precision_recall_curve,
    cohen_kappa_score, matthews_corrcoef
)

logger = logging.getLogger(__name__)


class ClinicalMetrics:
    """
    Calculate comprehensive clinical evaluation metrics.
    
    Attributes:
        y_true: Ground truth labels
        y_pred: Predicted labels (binary: 0 or 1)
        y_score: Prediction probabilities (for ROC/PR curves)
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize metrics calculator.
        
        Args:
            threshold: Classification threshold (0-1)
        """
        self.threshold = threshold
        self.metrics = {}
    
    def calculate_all(
        self,
        y_true: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        y_score: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate all clinical metrics.
        
        Medical Context:
        - Sensitivity (recall) is PRIMARY metric - minimize false negatives
        - Specificity is SECONDARY - false positives require review, not treatment
        - NPV is critical - confidence that negative prediction is correct
        - F2-score emphasizes recall over precision
        
        Args:
            y_true: True labels (0 or 1)
            y_pred: Binary predictions (0 or 1)
            y_score: Probability scores (0-1)
            
        Returns:
            Dictionary of all metrics
        """
        y_true = np.asarray(y_true).flatten()
        
        # Generate predictions from scores if not provided
        if y_pred is None and y_score is not None:
            y_pred = (y_score >= self.threshold).astype(int).flatten()
        elif y_pred is None:
            raise ValueError("Must provide y_pred or y_score")
        else:
            y_pred = np.asarray(y_pred).astype(int).flatten()
        
        # Validate dimensions
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})")
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate metrics
        self.metrics = {
            # Binary classification basics
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),  # Same as sensitivity
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'f2_score': f2_score(y_true, y_pred, zero_division=0),  # Emphasizes recall
            
            # Clinical metrics (most important!)
            'sensitivity': recall_score(y_true, y_pred, zero_division=0),  # True positive rate
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,  # True negative rate
            'ppv': precision_score(y_true, y_pred, zero_division=0),  # Positive predictive value
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative predictive value (CRITICAL)
            
            # Advanced metrics
            'kappa': cohen_kappa_score(y_true, y_pred),
            'mcc': matthews_corrcoef(y_true, y_pred),  # Matthews Correlation Coefficient
            
            # Confusion matrix components
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
        }
        
        # Add AUC if scores provided
        if y_score is not None:
            y_score = np.asarray(y_score).flatten()
            try:
                self.metrics['auc_roc'] = roc_auc_score(y_true, y_score)
            except ValueError:
                logger.warning("Could not calculate AUC (need both classes in data)")
                self.metrics['auc_roc'] = np.nan
        
        return self.metrics
    
    def get_roc_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get ROC curve points.
        
        Args:
            y_true: True labels
            y_score: Probability scores
            
        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        return roc_curve(y_true, y_score)
    
    def get_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get precision-recall curve points.
        
        Medical Context:
        - More informative than ROC for imbalanced datasets
        - Shows trade-off between precision and recall
        
        Args:
            y_true: True labels
            y_score: Probability scores
            
        Returns:
            Tuple of (precision, recall, thresholds)
        """
        return precision_recall_curve(y_true, y_score)
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        objective: str = 'sensitivity'
    ) -> Tuple[float, Dict]:
        """
        Find optimal decision threshold.
        
        Medical Context:
        - Default 0.5 threshold may not be optimal for clinical use
        - Can optimize for sensitivity (minimize false negatives)
        - Can optimize for specificity (minimize false positives)
        - Can optimize for balanced metrics
        
        Args:
            y_true: True labels
            y_score: Probability scores
            objective: 'sensitivity', 'specificity', 'balanced', 'f1', or 'f2'
            
        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        
        if objective == 'sensitivity':
            # Maximize sensitivity (minimize false negatives) - CLINICAL PRIORITY
            best_idx = np.argmax(tpr)
            
        elif objective == 'specificity':
            # Maximize specificity (minimize false positives)
            specificity = 1 - fpr
            best_idx = np.argmax(specificity)
            
        elif objective == 'balanced':
            # Youden's J statistic: maximize (sensitivity + specificity - 1)
            j_scores = tpr - fpr
            best_idx = np.argmax(j_scores)
            
        elif objective in ['f1', 'f2']:
            # Search for best F-score
            best_score = -1
            best_idx = 0
            for idx, thresh in enumerate(thresholds):
                y_pred = (y_score >= thresh).astype(int)
                if objective == 'f1':
                    score = f1_score(y_true, y_pred, zero_division=0)
                else:  # f2
                    score = f2_score(y_true, y_pred, zero_division=0)
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        optimal_threshold = thresholds[best_idx]
        
        # Calculate metrics at this threshold
        y_pred = (y_score >= optimal_threshold).astype(int)
        metrics_at_threshold = self.calculate_all(y_true, y_pred, y_score)
        
        logger.info(
            f"Optimal threshold ({objective}): {optimal_threshold:.3f} "
            f"Sensitivity: {metrics_at_threshold['sensitivity']:.3f}, "
            f"Specificity: {metrics_at_threshold['specificity']:.3f}"
        )
        
        return optimal_threshold, metrics_at_threshold
    
    def get_metrics_summary(self) -> str:
        """Get text summary of metrics."""
        if not self.metrics:
            return "No metrics calculated yet"
        
        summary = "Clinical Metrics Summary\n"
        summary += "=" * 50 + "\n"
        summary += f"Sensitivity (Recall):  {self.metrics['sensitivity']:.4f}  ⭐ CRITICAL\n"
        summary += f"Specificity:           {self.metrics['specificity']:.4f}\n"
        summary += f"NPV (Negative Pred):   {self.metrics['npv']:.4f}  ⭐ IMPORTANT\n"
        summary += f"PPV (Positive Pred):   {self.metrics['ppv']:.4f}\n"
        summary += f"Accuracy:              {self.metrics['accuracy']:.4f}\n"
        summary += f"F1-Score:              {self.metrics['f1_score']:.4f}\n"
        summary += f"F2-Score:              {self.metrics['f2_score']:.4f}\n"
        summary += f"AUC-ROC:               {self.metrics.get('auc_roc', 'N/A')}\n"
        summary += f"\nConfusion Matrix:\n"
        summary += f"  TP (True Positive):   {self.metrics['tp']}\n"
        summary += f"  TN (True Negative):   {self.metrics['tn']}\n"
        summary += f"  FP (False Positive):  {self.metrics['fp']}\n"
        summary += f"  FN (False Negative):  {self.metrics['fn']} ⚠️ MINIMIZE\n"
        
        return summary
    
    def __repr__(self) -> str:
        return f"ClinicalMetrics(threshold={self.threshold})"


if __name__ == "__main__":
    # Example usage
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 0])
    y_score = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.4, 0.85, 0.15, 0.2])
    
    metrics = ClinicalMetrics()
    results = metrics.calculate_all(y_true, y_pred, y_score)
    print(metrics.get_metrics_summary())
