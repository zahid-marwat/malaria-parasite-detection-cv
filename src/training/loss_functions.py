"""
Loss functions for medical malaria detection.

Specialized loss functions that prioritize sensitivity (high recall)
and handle class imbalance appropriately.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np


def weighted_binary_crossentropy(y_true, y_pred, pos_weight=1.0):
    """
    Binary crossentropy with class weighting.
    
    Medical Context:
    - Assigns higher weight to positive (infected) class
    - Prevents model from ignoring rare infected samples
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        pos_weight: Weight for positive class (> 1 emphasizes recall)
        
    Returns:
        Weighted binary crossentropy loss
    """
    return K.mean(K.binary_crossentropy(y_true, y_pred) * 
                  (y_true * pos_weight + (1 - y_true)))


class WeightedBinaryCrossentropy(keras.losses.Loss):
    """Weighted binary crossentropy loss class."""
    
    def __init__(self, pos_weight=1.0, name='weighted_bce'):
        super().__init__(name=name)
        self.pos_weight = pos_weight
    
    def call(self, y_true, y_pred):
        return K.mean(K.binary_crossentropy(y_true, y_pred) * 
                     (y_true * self.pos_weight + (1 - y_true)))


def focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal loss for handling class imbalance.
    
    Medical Context:
    - Focuses on hard negative examples (false positives)
    - Reduces contribution of easy negatives (true negatives)
    - Helps model learn difficult cases
    
    Args:
        alpha: Weighting factor
        gamma: Focusing parameter
        
    Returns:
        Focal loss function
    """
    def focal_crossentropy(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        # Calculate binary crossentropy
        bce = K.binary_crossentropy(y_true, y_pred)
        
        # Calculate focal weight
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = K.pow(1 - p_t, gamma)
        
        # Apply focal loss
        focal_loss_value = focal_weight * bce
        
        return K.mean(focal_loss_value)
    
    return focal_crossentropy


class FocalLoss(keras.losses.Loss):
    """Focal loss class."""
    
    def __init__(self, alpha=0.25, gamma=2.0, name='focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        bce = K.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = K.pow(1 - p_t, self.gamma)
        
        return K.mean(focal_weight * bce)


def sensitivity_specificity_loss(sensitivity_weight=0.7):
    """
    Loss function emphasizing sensitivity (recall).
    
    Medical Context:
    - Missing infected cases is more costly than false positives
    - Balances sensitivity and specificity with focus on recall
    
    Args:
        sensitivity_weight: Weight for sensitivity (0-1)
        
    Returns:
        Loss function
    """
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        # True positives and false negatives
        tp = K.sum(y_true * y_pred)
        fn = K.sum(y_true * (1 - y_pred))
        fp = K.sum((1 - y_true) * y_pred)
        tn = K.sum((1 - y_true) * (1 - y_pred))
        
        sensitivity = tp / (tp + fn + epsilon)
        specificity = tn / (tn + fp + epsilon)
        
        # Combined loss: prioritize sensitivity
        loss_value = (1 - sensitivity_weight) * (1 - sensitivity) + \
                     sensitivity_weight * (1 - specificity)
        
        return loss_value
    
    return loss


def calculate_class_weight(y_labels: np.ndarray) -> dict:
    """
    Calculate class weights for imbalanced dataset.
    
    Medical Context:
    - Malaria datasets often have more negatives than positives
    - Class weighting balances the influence of each class
    
    Args:
        y_labels: Array of class labels
        
    Returns:
        Dictionary mapping class indices to weights
    """
    unique_classes = np.unique(y_labels)
    class_counts = np.bincount(y_labels.astype(int))
    total_samples = len(y_labels)
    
    class_weights = {}
    for cls in unique_classes:
        # Weight inversely proportional to class frequency
        weight = total_samples / (len(unique_classes) * class_counts[cls])
        class_weights[int(cls)] = weight
    
    return class_weights


def get_loss_function(loss_name: str, **kwargs):
    """
    Get loss function by name.
    
    Args:
        loss_name: Name of loss function
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function
    """
    loss_functions = {
        'binary_crossentropy': keras.losses.BinaryCrossentropy(),
        'categorical_crossentropy': keras.losses.CategoricalCrossentropy(),
        'weighted_bce': WeightedBinaryCrossentropy(**kwargs),
        'focal_loss': FocalLoss(**kwargs),
        'sensitivity_specificity': sensitivity_specificity_loss(**kwargs),
    }
    
    if loss_name in loss_functions:
        return loss_functions[loss_name]
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


if __name__ == "__main__":
    # Test loss functions
    import tensorflow as tf
    
    y_true = tf.constant([[1.], [0.], [1.], [0.]])
    y_pred = tf.constant([[0.9], [0.1], [0.7], [0.2]])
    
    focal = FocalLoss()
    print(f"Focal loss: {focal(y_true, y_pred).numpy()}")
    
    weighted = WeightedBinaryCrossentropy(pos_weight=2.0)
    print(f"Weighted BCE: {weighted(y_true, y_pred).numpy()}")
