"""
Visualization utilities for medical imaging evaluation.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

logger = logging.getLogger(__name__)


class ClinicalVisualizer:
    """Generate clinical evaluation visualizations."""
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Plot confusion matrix with clinical labels.
        
        Medical Context:
        - False Negatives (missing infected) are most critical
        - False Positives trigger additional review but not treatment
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Uninfected', 'Infected'],
            yticklabels=['Uninfected', 'Infected'],
            ax=ax
        )
        
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Confusion Matrix - Malaria Detection\n(FN = missed infections ⚠️)')
        
        # Annotate with clinical meaning
        tn, fp, fn, tp = cm.ravel()
        textstr = f'TP: {tp} | FP: {fp}\nFN: {fn} (⚠️ Critical) | TN: {tn}'
        ax.text(0.5, -0.3, textstr, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray,
        y_score: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """Plot ROC curve."""
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate (Sensitivity)')
        ax.set_title('ROC Curve - Malaria Detection')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_precision_recall_curve(
        y_true: np.ndarray,
        y_score: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """Plot precision-recall curve (more informative for imbalanced data)."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(recall, precision, color='darkorange', lw=2, label=f'AUC = {pr_auc:.3f}')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall (Sensitivity)')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve - Malaria Detection')
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR curve saved to {save_path}")
        
        return fig


if __name__ == "__main__":
    pass
