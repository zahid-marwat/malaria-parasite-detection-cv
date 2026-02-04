"""
Model interpretability for malaria detection.

Provides Grad-CAM visualizations and other interpretability techniques
to understand what the model focuses on.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Maps) for model interpretation.
    
    Generates heatmaps showing which regions the model focuses on for predictions.
    """
    
    def __init__(self, model: keras.Model, layer_name: Optional[str] = None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Keras model
            layer_name: Name of layer to use (default: last conv layer)
        """
        self.model = model
        
        # Find layer name if not provided
        if layer_name is None:
            # Find last convolutional layer
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break
        
        self.layer_name = layer_name or model.layers[-2].name
        logger.info(f"Using layer '{self.layer_name}' for Grad-CAM")
    
    def generate_heatmap(
        self,
        image: np.ndarray,
        class_index: int = 1
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for single image.
        
        Args:
            image: Input image (H×W×3)
            class_index: Class to generate heatmap for
            
        Returns:
            Heatmap (H×W)
        """
        # Create model that outputs both predictions and last conv layer outputs
        grad_model = keras.Model(
            inputs=self.model.input,
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )
        
        # Record gradient computation
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image[np.newaxis, :])
            if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
                # Multi-class
                class_channel = predictions[:, class_index]
            else:
                # Binary
                class_channel = predictions[:, 0]
        
        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight conv layer outputs by gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        
        return heatmap.numpy()
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            image: Original image (H×W×3), values in [0, 1] or [0, 255]
            heatmap: Grad-CAM heatmap (H×W), values in [0, 1]
            alpha: Blending alpha
            colormap: OpenCV colormap name
            
        Returns:
            Image with heatmap overlay (H×W×3)
        """
        # Normalize image if needed
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        # Resize heatmap to match image size
        if heatmap.shape != image_uint8.shape[:2]:
            heatmap = cv2.resize(heatmap, (image_uint8.shape[1], image_uint8.shape[0]))
        
        # Convert heatmap to uint8
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        
        # Apply colormap
        if colormap == 'jet':
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        elif colormap == 'hot':
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_HOT)
        elif colormap == 'viridis':
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_VIRIDIS)
        else:
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Convert BGR to RGB
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend images
        overlay = cv2.addWeighted(image_uint8, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay


class SaliencyMap:
    """
    Generate saliency maps showing pixel importance.
    """
    
    def __init__(self, model: keras.Model):
        self.model = model
    
    def generate_saliency(
        self,
        image: np.ndarray,
        class_index: int = 1
    ) -> np.ndarray:
        """
        Generate saliency map for image.
        
        Args:
            image: Input image (H×W×3)
            class_index: Class to generate saliency for
            
        Returns:
            Saliency map (H×W)
        """
        image_input = tf.Variable(image[np.newaxis, :].astype(tf.float32))
        
        with tf.GradientTape() as tape:
            predictions = self.model(image_input)
            if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
                target = predictions[:, class_index]
            else:
                target = predictions[:, 0]
        
        # Compute gradients
        grads = tape.gradient(target, image_input)
        
        # Take maximum absolute value across channels
        saliency = tf.reduce_max(tf.abs(grads), axis=-1)
        saliency = tf.squeeze(saliency)
        
        # Normalize
        saliency = saliency / (tf.reduce_max(saliency) + 1e-10)
        
        return saliency.numpy()


if __name__ == "__main__":
    pass
