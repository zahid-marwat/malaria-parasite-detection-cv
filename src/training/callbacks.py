"""
Custom callbacks for medical model training.

Monitors clinical metrics (sensitivity, specificity) rather than just accuracy.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Callable

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import sensitivity_score, specificity_score, auc, roc_curve

logger = logging.getLogger(__name__)


class ClinicalMetricsCallback(keras.callbacks.Callback):
    """
    Monitor clinical metrics during training.
    
    Medical Context:
    - Tracks sensitivity (recall) - CRITICAL for malaria detection
    - Tracks specificity - important but secondary to sensitivity
    - Enables stopping based on clinical performance
    """
    
    def __init__(
        self,
        validation_data: tuple,
        metric_name: str = 'sensitivity',
        save_best_threshold: bool = False
    ):
        """
        Initialize callback.
        
        Args:
            validation_data: Tuple of (X_val, y_val)
            metric_name: Metric to monitor ('sensitivity', 'specificity', 'auc')
            save_best_threshold: Save optimal decision threshold
        """
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.metric_name = metric_name
        self.save_best_threshold = save_best_threshold
        self.best_threshold = 0.5
        self.history = {metric_name: []}
    
    def on_epoch_end(self, epoch, logs=None):
        """Calculate clinical metrics at end of each epoch."""
        logs = logs or {}
        
        # Get predictions
        y_pred_proba = self.model.predict(self.X_val, verbose=0)
        
        # Try threshold of 0.5 first
        y_pred = (y_pred_proba >= 0.5).astype(int).flatten()
        y_val_flat = self.y_val.flatten()
        
        # Calculate metrics
        sensitivity = sensitivity_score(y_val_flat, y_pred)
        specificity = specificity_score(y_val_flat, y_pred)
        
        logs[f'val_{self.metric_name}'] = sensitivity if self.metric_name == 'sensitivity' else specificity
        
        # Log
        logger.info(
            f"Epoch {epoch+1} - "
            f"val_sensitivity: {sensitivity:.4f}, "
            f"val_specificity: {specificity:.4f}"
        )
        
        self.history[self.metric_name].append(sensitivity)
    
    def get_best_threshold(self) -> float:
        """Find optimal decision threshold."""
        y_pred_proba = self.model.predict(self.X_val, verbose=0).flatten()
        y_val_flat = self.y_val.flatten()
        
        fpr, tpr, thresholds = roc_curve(y_val_flat, y_pred_proba)
        
        # Youden's J statistic
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        
        self.best_threshold = thresholds[best_idx]
        return self.best_threshold


class SensitivityEarlyStoppingCallback(keras.callbacks.Callback):
    """
    Early stopping based on sensitivity (recall).
    
    Medical Context:
    - Stops training if sensitivity plateaus or decreases
    - Prevents overfitting while maintaining high recall
    """
    
    def __init__(
        self,
        validation_data: tuple,
        patience: int = 15,
        min_sensitivity: float = 0.85,
        restore_best: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            validation_data: Tuple of (X_val, y_val)
            patience: Number of epochs with no improvement before stopping
            min_sensitivity: Minimum acceptable sensitivity
            restore_best: Restore weights from epoch with best sensitivity
        """
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.patience = patience
        self.min_sensitivity = min_sensitivity
        self.restore_best = restore_best
        self.wait_count = 0
        self.best_sensitivity = 0.0
        self.best_weights = None
        self.best_epoch = 0
    
    def on_epoch_end(self, epoch, logs=None):
        """Check sensitivity at end of epoch."""
        logs = logs or {}
        
        y_pred_proba = self.model.predict(self.X_val, verbose=0)
        y_pred = (y_pred_proba >= 0.5).astype(int).flatten()
        y_val_flat = self.y_val.flatten()
        
        sensitivity = sensitivity_score(y_val_flat, y_pred)
        
        if sensitivity > self.best_sensitivity:
            self.best_sensitivity = sensitivity
            self.wait_count = 0
            self.best_weights = [w.numpy() for w in self.model.weights]
            self.best_epoch = epoch
            
            logger.info(f"New best sensitivity: {sensitivity:.4f}")
        else:
            self.wait_count += 1
        
        if self.wait_count >= self.patience:
            logger.info(
                f"Stopping training: sensitivity no improvement for {self.patience} epochs"
            )
            if self.restore_best:
                logger.info(f"Restoring weights from epoch {self.best_epoch}")
                for w, best_w in zip(self.model.weights, self.best_weights):
                    w.assign(best_w)
            self.model.stop_training = True


class ModelCheckpointClinical(keras.callbacks.Callback):
    """
    Save model based on clinical metrics.
    
    Medical Context:
    - Saves best model based on sensitivity or specificity
    - Ensures we keep clinically validated model
    """
    
    def __init__(
        self,
        validation_data: tuple,
        save_dir: str = 'models/',
        metric: str = 'sensitivity',
        mode: str = 'max',
        save_freq: int = 1
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            validation_data: Tuple of (X_val, y_val)
            save_dir: Directory to save models
            metric: Metric to monitor ('sensitivity', 'specificity')
            mode: 'max' or 'min'
            save_freq: Save every N epochs
        """
        super().__init__()
        self.X_val, self.y_val = validation_data
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self.mode = mode
        self.save_freq = save_freq
        self.best_value = -np.inf if mode == 'max' else np.inf
        self.last_saved = -np.inf
    
    def on_epoch_end(self, epoch, logs=None):
        """Save model if improved."""
        if epoch % self.save_freq != 0:
            return
        
        y_pred_proba = self.model.predict(self.X_val, verbose=0)
        y_pred = (y_pred_proba >= 0.5).astype(int).flatten()
        y_val_flat = self.y_val.flatten()
        
        if self.metric == 'sensitivity':
            metric_value = sensitivity_score(y_val_flat, y_pred)
        else:
            metric_value = specificity_score(y_val_flat, y_pred)
        
        is_improvement = (
            (self.mode == 'max' and metric_value > self.best_value) or
            (self.mode == 'min' and metric_value < self.best_value)
        )
        
        if is_improvement:
            self.best_value = metric_value
            model_path = self.save_dir / f"best_model_{self.metric}_{metric_value:.4f}.h5"
            self.model.save(str(model_path))
            logger.info(f"Model saved: {model_path}")


if __name__ == "__main__":
    pass
