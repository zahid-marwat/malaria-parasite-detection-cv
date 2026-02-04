"""
Model training pipeline for malaria parasite detection.

Implements complete training loop with class balancing, early stopping,
learning rate scheduling, and clinical metric monitoring.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.training.config import TrainingConfig
from src.training.loss_functions import calculate_class_weight, get_loss_function
from src.training.callbacks import ClinicalMetricsCallback, SensitivityEarlyStoppingCallback

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Training pipeline with medical-appropriate metrics and callbacks.
    """
    
    def __init__(
        self,
        model: keras.Model,
        config: Union[TrainingConfig, str, Path],
        model_save_dir: Union[str, Path] = 'models/',
        log_dir: Union[str, Path] = 'tb_logs/'
    ):
        """
        Initialize trainer.
        
        Args:
            model: Keras model to train
            config: TrainingConfig object or path to config YAML
            model_save_dir: Directory to save trained models
            log_dir: Directory for TensorBoard logs
        """
        self.model = model
        
        # Load config if path provided
        if isinstance(config, (str, Path)):
            self.config = TrainingConfig.from_yaml(str(config))
        else:
            self.config = config
        
        self.model_save_dir = Path(model_save_dir)
        self.log_dir = Path(log_dir)
        
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = None
        self.training_stats = {}
        
        logger.info(f"Initialized ModelTrainer with config: {self.config}")
    
    def train(
        self,
        train_images: np.ndarray,
        train_labels: np.ndarray,
        val_images: np.ndarray,
        val_labels: np.ndarray,
        class_weights: Optional[Dict] = None,
        verbose: int = 1
    ) -> Dict:
        """
        Train model with medical-appropriate monitoring.
        
        Medical Considerations:
        - Class weighting handles imbalanced data
        - Early stopping monitors sensitivity (recall), not accuracy
        - Learning rate scheduling prevents overshooting
        - All metrics logged for clinical review
        
        Args:
            train_images: Training images (N×H×W×3)
            train_labels: Training labels (N,)
            val_images: Validation images
            val_labels: Validation labels
            class_weights: Optional custom class weights
            verbose: Verbosity level
            
        Returns:
            Dictionary with training history and statistics
        """
        logger.info("Starting training...")
        
        # Compile model if not already compiled
        if not self.model.optimizer:
            self._compile_model()
        
        # Calculate class weights if needed
        if self.config.use_class_weights and class_weights is None:
            class_weights = calculate_class_weight(train_labels)
            logger.info(f"Calculated class weights: {class_weights}")
        
        # Prepare callbacks
        callbacks = self._setup_callbacks(val_images, val_labels)
        
        # Reshape labels if needed
        if len(train_labels.shape) == 1:
            train_labels = train_labels.reshape(-1, 1)
        if len(val_labels.shape) == 1:
            val_labels = val_labels.reshape(-1, 1)
        
        # Train model
        self.history = self.model.fit(
            train_images,
            train_labels,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=(val_images, val_labels),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=verbose
        )
        
        # Save training statistics
        self._save_training_stats()
        
        logger.info("Training complete")
        
        return self._get_training_summary()
    
    def _compile_model(self) -> None:
        """Compile model with medical-appropriate settings."""
        optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        
        loss_fn = get_loss_function(
            self.config.loss_function,
            pos_weight=2.0 if 'weighted' in self.config.loss_function else 1.0
        )
        
        metrics = [
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with loss={self.config.loss_function}")
    
    def _setup_callbacks(
        self,
        val_images: np.ndarray,
        val_labels: np.ndarray
    ) -> list:
        """Set up training callbacks."""
        callbacks = []
        
        # TensorBoard callback
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=str(self.log_dir),
                histogram_freq=1,
                write_graph=True
            )
        )
        
        # Learning rate scheduling
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        )
        
        # Clinical metrics callback
        callbacks.append(
            ClinicalMetricsCallback(
                validation_data=(val_images, val_labels),
                metric_name='sensitivity'
            )
        )
        
        # Early stopping (based on sensitivity, not accuracy!)
        callbacks.append(
            SensitivityEarlyStoppingCallback(
                validation_data=(val_images, val_labels),
                patience=self.config.early_stopping_patience,
                restore_best=True
            )
        )
        
        # Model checkpoint
        model_checkpoint_path = self.model_save_dir / 'checkpoint_model.h5'
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(model_checkpoint_path),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        )
        
        return callbacks
    
    def _save_training_stats(self) -> None:
        """Save training statistics."""
        stats = {
            'config': self.config.to_dict(),
            'total_epochs': len(self.history.history['loss']),
            'final_train_loss': float(self.history.history['loss'][-1]),
            'final_val_loss': float(self.history.history['val_loss'][-1]),
            'final_accuracy': float(self.history.history['accuracy'][-1]),
            'final_val_accuracy': float(self.history.history['val_accuracy'][-1]),
        }
        
        self.training_stats = stats
        
        # Save to JSON
        stats_path = self.model_save_dir / 'training_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Training stats saved to {stats_path}")
    
    def _get_training_summary(self) -> Dict:
        """Get training summary."""
        return {
            'history': self.history.history,
            'stats': self.training_stats
        }
    
    def save_model(self, name: str = 'trained_model', format: str = 'h5') -> None:
        """
        Save trained model.
        
        Args:
            name: Model name (without extension)
            format: 'h5' or 'tf'
        """
        if format == 'h5':
            save_path = self.model_save_dir / f"{name}.h5"
            self.model.save(str(save_path))
        elif format == 'tf':
            save_path = self.model_save_dir / name
            self.model.save(str(save_path))
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Model saved to {save_path}")
    
    def get_history(self) -> Dict:
        """Get training history."""
        if self.history is None:
            raise ValueError("No training history available. Train model first.")
        return self.history.history


if __name__ == "__main__":
    pass
