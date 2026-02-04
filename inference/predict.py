"""
Inference pipeline for malaria parasite detection.

Single-image and batch prediction with confidence scores.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image

from src.data.preprocessor import MicroscopyPreprocessor

logger = logging.getLogger(__name__)


class MalariaDiagnosticPredictor:
    """
    Make malaria predictions on blood smear images.
    
    Provides confidence scores and diagnostic information.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        preprocessor: Optional[MicroscopyPreprocessor] = None,
        threshold: float = 0.5
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model
            preprocessor: Optional preprocessor (default: standard)
            threshold: Decision threshold (0-1)
        """
        self.model = keras.models.load_model(str(model_path))
        self.preprocessor = preprocessor or MicroscopyPreprocessor(target_size=(224, 224))
        self.threshold = threshold
        
        logger.info(f"Loaded model from {model_path}")
    
    def predict(
        self,
        image_path: Union[str, Path],
        return_heatmap: bool = False
    ) -> Dict:
        """
        Predict on single image.
        
        Medical Context:
        - Returns binary prediction: infected/uninfected
        - Provides confidence score
        - Can return attention heatmap
        
        Args:
            image_path: Path to blood smear image
            return_heatmap: Whether to return attention heatmap
            
        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess image
        image = self.preprocessor.preprocess_image(str(image_path))
        image_batch = np.expand_dims(image, axis=0)
        
        # Make prediction
        prediction = self.model.predict(image_batch, verbose=0)[0]
        
        if len(prediction) == 1:
            # Binary output
            confidence = float(prediction[0])
            label = 'Infected' if confidence >= self.threshold else 'Uninfected'
            infected_prob = confidence
            uninfected_prob = 1 - confidence
        else:
            # Multi-class
            class_idx = np.argmax(prediction)
            confidence = float(prediction[class_idx])
            label = 'Infected' if class_idx == 1 else 'Uninfected'
            infected_prob = float(prediction[1]) if len(prediction) > 1 else 0
            uninfected_prob = float(prediction[0])
        
        result = {
            'image_path': str(image_path),
            'label': label,
            'confidence': confidence,
            'infected_probability': infected_prob,
            'uninfected_probability': uninfected_prob,
            'threshold': self.threshold,
            'raw_prediction': prediction.tolist()
        }
        
        logger.info(f"Prediction: {label} (confidence: {confidence:.2%})")
        
        return result
    
    def predict_batch(
        self,
        image_dir: Union[str, Path],
        file_extension: str = '*.jpg'
    ) -> list:
        """
        Predict on batch of images.
        
        Args:
            image_dir: Directory containing images
            file_extension: File pattern to match
            
        Returns:
            List of prediction results
        """
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob(file_extension))
        
        logger.info(f"Found {len(image_files)} images to predict")
        
        results = []
        for img_path in image_files:
            try:
                result = self.predict(img_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting {img_path}: {e}")
                continue
        
        return results


if __name__ == "__main__":
    pass
